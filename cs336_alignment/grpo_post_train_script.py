from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import os
import json
import torch
from transformers import PreTrainedTokenizerBase
from cs336_alignment.util import (
    tokenize_prompt_and_output,
    get_response_log_probs,
    sft_microbatch_train_step,
    grpo_microbatch_train_step,
    compute_group_normalized_rewards,
)
from cs336_alignment.gsm_benchmark_script import evaluate_gsm
from cs336_alignment.sft_post_train_script import (
    load_gsm8k_train_data,
)
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
import logging
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from cs336_alignment.gsm_benchmark_script import evaluate_gsm
from vllm import LLM, SamplingParams
from peft import LoraConfig, get_peft_model
from typing import Any
import torch.nn.functional as F
import copy
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict
from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger(__name__)


class MetricsTracker:
    """Tracks scalar metrics over training and exports them as figures."""

    def __init__(self, log_file: str | None = None, tb_log_dir: str | None = None):
        self.metrics: dict[str, list[tuple[int, float]]] = defaultdict(list)
        self._log_file = log_file
        self._tb_writer: SummaryWriter | None = None
        if log_file:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            with open(log_file, "w") as f:
                pass
        if tb_log_dir:
            self._tb_writer = SummaryWriter(log_dir=tb_log_dir)

    def log(self, name: str, step: int, value: float):
        self.metrics[name].append((step, value))
        if self._log_file:
            with open(self._log_file, "a") as f:
                f.write(
                    json.dumps({"metric": name, "step": step, "value": value}) + "\n"
                )
        if self._tb_writer:
            self._tb_writer.add_scalar(name, value, step)

    def export_figures(self, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        for name, values in self.metrics.items():
            steps, vals = zip(*values)
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(steps, vals)
            ax.set_xlabel("Rollout Step")
            ax.set_ylabel(name)
            ax.set_title(name)
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            safe_name = name.replace("/", "_").replace(" ", "_")
            fig.savefig(os.path.join(output_dir, f"{safe_name}.png"), dpi=150)
            plt.close(fig)
        # combined figure
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        ordered = ["loss", "train_reward", "val_reward", "clip_fraction"]
        for ax, key in zip(axes, ordered):
            if key in self.metrics:
                steps, vals = zip(*self.metrics[key])
                ax.plot(steps, vals)
                ax.set_title(key)
                ax.set_xlabel("Rollout Step")
                ax.grid(True, alpha=0.3)
            else:
                ax.set_visible(False)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "metrics_combined.png"), dpi=150)
        plt.close(fig)

    def close(self):
        if self._tb_writer:
            self._tb_writer.close()


def get_rollout_samping_param(n_rollout: int) -> SamplingParams:
    sampling_params = SamplingParams(
        temperature=1.0,
        n=n_rollout,
        max_tokens=1024,
        min_tokens=4,
    )
    sampling_params.stop = ["</answer>"]
    sampling_params.include_stop_str_in_output = True

    return sampling_params


def grpo_rollout_batch_training_loop(
    model: torch.nn.Module,
    train_batch_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    gradient_accumulation_steps: int,
    cliprange: float,
    n_epoch: int,
    step_idx: int,
    update_idx: int,
    use_async_grpo: bool = False,
    async_grpo_apply_rollout_importance_sampling: bool = False,
) -> tuple[int, int, dict[str, float]]:
    losses = []
    clip_fractions = []
    for epoch_idx in range(n_epoch):
        optimizer.zero_grad()
        if use_async_grpo:
            train_batches = []
            model.eval()
            for train_batch in train_batch_loader:
                input_ids = train_batch["input_ids"].to(device)
                labels = train_batch["labels"].to(device)
                log_probs_and_entropy = get_response_log_probs(
                    model, input_ids, labels, True
                )
                old_log_probs = log_probs_and_entropy["log_probs"]  # type: ignore
                if async_grpo_apply_rollout_importance_sampling:
                    rollout_log_probs = train_batch["old_log_probs"].to(device)
                    importance_weights = torch.exp(old_log_probs - rollout_log_probs)
                    train_batch["importance_weights"] = importance_weights.cpu()
                train_batch["old_log_probs"] = old_log_probs.cpu()  # type: ignore
                train_batches.append(train_batch)
            model.train()
        else:
            train_batches = train_batch_loader

        for train_batch in train_batches:
            input_ids = train_batch["input_ids"].to(device)
            labels = train_batch["labels"].to(device)
            response_mask = train_batch["response_mask"].to(device)
            old_log_probs = train_batch["old_log_probs"].to(device)
            advantages = train_batch["advantages"].to(device)
            importance_weights = train_batch.get("importance_weights", None)
            if importance_weights is not None:
                importance_weights = importance_weights.to(device)
            log_probs_and_entropy = get_response_log_probs(
                model, input_ids, labels, True
            )
            loss, step_data = grpo_microbatch_train_step(
                log_probs_and_entropy["log_probs"],  # type: ignore
                response_mask,
                gradient_accumulation_steps,
                "grpo_clip",
                None,  # raw reward
                advantages,
                old_log_probs,
                cliprange,
            )
            losses.append(loss.item())
            if "clip_fraction" in step_data:
                clip_fractions.append(step_data["clip_fraction"].item())
            if (step_idx + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                if update_idx % 10 == 0:
                    logger.info(
                        "epoch %d step %d loss %.4f, entropy %.4f",
                        epoch_idx,
                        step_idx,
                        loss.item(),
                        log_probs_and_entropy["token_entropy"].mean().item(),  # type: ignore
                    )
                update_idx += 1
            step_idx += 1
    loop_metrics = {
        "mean_loss": sum(losses) / len(losses) if losses else 0.0,
        "mean_clip_fraction": (
            sum(clip_fractions) / len(clip_fractions) if clip_fractions else 0.0
        ),
    }
    return step_idx, update_idx, loop_metrics


def rollout(
    model: LLM, data: list[dict[str, str]], sampling_param: SamplingParams
) -> list[dict[str, Any]]:
    prompt_file = os.path.join(os.path.dirname(__file__), "prompts", "r1_zero.prompt")

    with open(prompt_file, "r") as f:
        prompt_template = f.read()

    prompt_list = [
        prompt_template.format(question=sample["question"]) for sample in data
    ]

    responses = model.generate(prompt_list, sampling_param)
    rollout_data = []
    for sample, response in zip(data, responses):
        rollout_data.append(
            {
                "prompt": prompt_template.format(question=sample["question"]),
                "outputs": [output.text for output in response.outputs],
                "answer": sample["answer"],
            }
        )

    return rollout_data


def create_grpo_rollout_batch_dataloader(
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    device: str,
    rollout_data: list[dict[str, Any]],
    train_batch_size: int,
    advantage_eps: float,
    normalize_by_std: bool,
) -> tuple[DataLoader, dict[str, torch.Tensor]]:
    data_processed = []
    raw_rewards = []
    for sample in rollout_data:
        prompt = sample["prompt"]
        answer = sample["answer"]
        n_output = len(sample["outputs"])
        prompt_repeated = [prompt] * n_output
        tokenized_results = tokenize_prompt_and_output(
            prompt_repeated, sample["outputs"], tokenizer
        )
        with torch.no_grad():
            log_probs_and_entropy = get_response_log_probs(
                model,
                tokenized_results["input_ids"].to(device),
                tokenized_results["labels"].to(device),
                False,
            )
        grpo_reward, raw_reward, reward_metadata = compute_group_normalized_rewards(
            r1_zero_reward_fn,
            sample["outputs"],
            [answer] * n_output,
            n_output,
            advantage_eps,
            normalize_by_std,
        )
        raw_rewards.append(raw_reward)
        for i in range(n_output):
            output = sample["outputs"][i]
            adv = grpo_reward[i].cpu().item()
            old_log_probs = log_probs_and_entropy["log_probs"][i, :].cpu()  # type: ignore
            data_processed.append(
                {
                    "prompt": prompt,
                    "output": output,
                    "old_log_probs": old_log_probs,
                    "advantage": adv,
                }
            )

    raw_reward_mean = torch.cat(raw_rewards, dim=0).mean()

    def collate_fn(batch):
        tokenized_results = tokenize_prompt_and_output(
            [s["prompt"] for s in batch],
            [s["output"] for s in batch],
            tokenizer,
        )
        # collate log probs
        old_log_probs_list = []
        for s in batch:
            old_log_probs = s["old_log_probs"]
            pad_len = tokenized_results["labels"].shape[1] - old_log_probs.shape[0]
            if pad_len <= 0:
                old_log_probs = old_log_probs[: tokenized_results["labels"].shape[1]]
            else:
                old_log_probs = F.pad(old_log_probs, (0, pad_len))
            old_log_probs_list.append(old_log_probs)
        tokenized_results["old_log_probs"] = torch.stack(old_log_probs_list, dim=0)
        # advantage
        advs_tensor = torch.tensor([s["advantage"] for s in batch])
        tokenized_results["advantages"] = advs_tensor[:, None]

        return tokenized_results

    train_batch_loader = DataLoader(
        data_processed,  # type: ignore
        batch_size=train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

    return train_batch_loader, {"raw_reward_mean": raw_reward_mean}


def train_script(
    model_name: str = "Qwen/Qwen2.5-Math-1.5B",
    expt_name: str = "Qwen2.5_Math_1.5B_GRPO",
    n_grpo_steps: int = 200,
    rollout_batch_size: int = 256,
    group_size: int = 8,
    micro_batch_size: int = 16,
    gradient_accumulation_steps: int = 4,
    n_train_epoch_per_rollout: int = 1,
    lr: float = 1e-5,
    use_lora: bool = True,
    grpo_clip_range: float = 0.2,
    use_std_normalization=True,
    eval_every_n_step: int = 50,
    use_async_grpo: bool = False,
):
    advantage_eps: float = 1e-6
    assert (
        rollout_batch_size % group_size == 0
    ), "rollout_batch_size must be divisible by group_size"
    n_prompts_per_rollout_batch = rollout_batch_size // group_size

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("using device: %s", device)
    # train model; to be initialized later
    policy_model: torch.nn.Module | None = None
    tokenizer: PreTrainedTokenizerBase | None = None
    optimizer: torch.optim.Optimizer | None = None
    rollout_model_path: str = model_name

    # async grpo assert
    if use_async_grpo:
        assert (
            n_train_epoch_per_rollout > 1
        ), "async grpo only supports more than 1 train epoch per rollout batch"

    # define the train dataset
    data = load_gsm8k_train_data()
    assert n_prompts_per_rollout_batch <= len(data)
    rollout_batch_loader = DataLoader(
        data,  # type: ignore
        batch_size=n_prompts_per_rollout_batch,
        shuffle=True,
        drop_last=True,
        collate_fn=lambda batch: batch,
    )
    log_dir = os.path.join(os.path.dirname(__file__), "..", "train_output", expt_name)
    tb_dir = os.path.join(log_dir, "tb_logs")
    metrics = MetricsTracker(
        log_file=os.path.join(log_dir, "metrics.jsonl"),
        tb_log_dir=tb_dir,
    )
    rollout_step_idx = 0
    train_step_idx = 0
    train_update_idx = 0
    while rollout_step_idx < n_grpo_steps:
        for prompt_batch in rollout_batch_loader:
            # load the rollout model from local path
            vllm = LLM(rollout_model_path)
            sampling_param = get_rollout_samping_param(group_size)
            if rollout_step_idx % eval_every_n_step == 0:
                logger.info("Eval at %d, %d", rollout_step_idx, train_step_idx)
                eval_param = SamplingParams(temperature=1.0, top_p=1.0, max_tokens=1024)
                eval_param.stop = ["</answer>"]
                eval_param.include_stop_str_in_output = True
                val_reward = evaluate_gsm(expt_name, vllm, eval_param)
                metrics.log("val_reward", rollout_step_idx, val_reward)
                logger.info(
                    "Rollout step %d, val reward %.4f", rollout_step_idx, val_reward
                )
            rollout_data = rollout(vllm, prompt_batch, sampling_param)
            del vllm
            torch.cuda.empty_cache()
            # load the train model
            if policy_model is None:
                policy_model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.bfloat16,
                    attn_implementation="flash_attention_2",
                )
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                if use_lora:
                    lora_config = LoraConfig(
                        r=16,
                        lora_alpha=32,
                        target_modules=["q_proj", "v_proj"],
                        lora_dropout=0.05,
                        task_type="CAUSAL_LM",
                    )
                    policy_model = get_peft_model(policy_model, lora_config)  # type: ignore
                assert policy_model is not None
                policy_model.to(device)
                # optimizer
                optimizer = AdamW(policy_model.parameters(), lr=lr)
            else:
                # move the model to gpu
                policy_model.to(device)
            assert policy_model is not None
            assert tokenizer is not None
            assert optimizer is not None
            policy_model.eval()
            train_batch_dataloader, rollout_metadata = (
                create_grpo_rollout_batch_dataloader(
                    policy_model,
                    tokenizer,
                    device,
                    rollout_data,
                    micro_batch_size,
                    advantage_eps,
                    use_std_normalization,
                )
            )
            policy_model.train()
            train_reward = rollout_metadata["raw_reward_mean"].item()
            metrics.log("train_reward", rollout_step_idx, train_reward)
            if rollout_step_idx % 10 == 0:
                logger.info(
                    "Rollout step %d, train reward mean %.4f",
                    rollout_step_idx,
                    train_reward,
                )
            train_step_idx, train_update_idx, loop_metrics = (
                grpo_rollout_batch_training_loop(
                    policy_model,
                    train_batch_dataloader,
                    optimizer,
                    device,
                    gradient_accumulation_steps,
                    grpo_clip_range,
                    n_train_epoch_per_rollout,
                    train_step_idx,
                    train_update_idx,
                )
            )
            metrics.log("loss", rollout_step_idx, loop_metrics["mean_loss"])
            metrics.log(
                "clip_fraction", rollout_step_idx, loop_metrics["mean_clip_fraction"]
            )
            rollout_model_path = os.path.join(
                os.path.dirname(__file__),
                "..",
                "train_output",
                expt_name,
                f"grpo_step_{rollout_step_idx // 100}",  # save
            )
            os.makedirs(rollout_model_path, exist_ok=True)
            policy_model.cpu()
            if use_lora:
                merged = copy.deepcopy(policy_model).merge_and_unload()
                merged.save_pretrained(rollout_model_path)
                del merged
            else:
                policy_model.save_pretrained(rollout_model_path)
            tokenizer.save_pretrained(rollout_model_path)
            rollout_step_idx += 1

    # Export metric figures
    figures_dir = os.path.join(
        os.path.dirname(__file__), "..", "train_output", expt_name, "figures"
    )
    metrics.export_figures(figures_dir)
    metrics.close()


if __name__ == "__main__":
    log_dir = os.path.join(os.path.dirname(__file__), "..", "train_output")
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(filename)s:%(lineno)d %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        filename=os.path.join(log_dir, "train.log"),
        filemode="w",
    )
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, default="Qwen/Qwen2.5-Math-1.5B", dest="model_name"
    )
    parser.add_argument(
        "--expt", type=str, default="Qwen2.5_Math_1.5B_GRPO", dest="expt_name"
    )
    parser.add_argument(
        "--use_async_grpo",
        action=argparse.BooleanOptionalAction,
        default=False,
        dest="use_async_grpo",
    )
    parser.add_argument("--grpo_steps", type=int, default=200, dest="n_grpo_steps")
    parser.add_argument("--rollout_batch_size", type=int, default=256)
    parser.add_argument("--group_size", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=16, dest="micro_batch_size")
    parser.add_argument(
        "--accum_steps", type=int, default=4, dest="gradient_accumulation_steps"
    )
    parser.add_argument(
        "--train_epochs", type=int, default=1, dest="n_train_epoch_per_rollout"
    )
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--use_lora", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--clip_range", type=float, default=0.2, dest="grpo_clip_range")
    parser.add_argument(
        "--use_std_normalization", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--eval_every", type=int, default=50, dest="eval_every_n_step")
    args = parser.parse_args()
    train_script(
        model_name=args.model_name,
        expt_name=args.expt_name,
        n_grpo_steps=args.n_grpo_steps,
        rollout_batch_size=args.rollout_batch_size,
        group_size=args.group_size,
        micro_batch_size=args.micro_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        n_train_epoch_per_rollout=args.n_train_epoch_per_rollout,
        lr=args.lr,
        use_lora=args.use_lora,
        grpo_clip_range=args.grpo_clip_range,
        use_std_normalization=args.use_std_normalization,
        eval_every_n_step=args.eval_every_n_step,
    )
