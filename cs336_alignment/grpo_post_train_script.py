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

logger = logging.getLogger(__name__)


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
) -> tuple[int, int]:
    optimizer.zero_grad()
    for epoch_idx in range(n_epoch):
        for train_batch in train_batch_loader:
            input_ids = train_batch["input_ids"].to(device)
            labels = train_batch["labels"].to(device)
            response_mask = train_batch["response_mask"].to(device)
            old_log_probs = train_batch["old_log_probs"].to(device)
            advantages = train_batch["advantages"].to(device)
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
    return step_idx, update_idx


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
                evaluate_gsm(expt_name, vllm, eval_param)
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
            if rollout_step_idx % 10 == 0:
                logger.info(
                    "Rollout step %d, train reward mean %.4f",
                    rollout_step_idx,
                    rollout_metadata["raw_reward_mean"].item(),
                )
            train_step_idx, train_update_idx = grpo_rollout_batch_training_loop(
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


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(filename)s:%(lineno)d %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, default="Qwen/Qwen2.5-Math-1.5B", dest="model_name"
    )
    parser.add_argument(
        "--expt", type=str, default="Qwen2.5_Math_1.5B_GRPO", dest="expt_name"
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
    parser.add_argument("--lr", type=float, default=1e-5)
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
