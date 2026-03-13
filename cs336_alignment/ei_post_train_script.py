from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
import argparse
import os
import json
import torch
from cs336_alignment.util import (
    tokenize_prompt_and_output,
    get_response_log_probs,
    sft_microbatch_train_step,
)
import logging
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from cs336_alignment.gsm_benchmark_script import evaluate_gsm
from cs336_alignment.sft_post_train_script import (
    load_gsm8k_train_data,
    sft_training_loop,
)
from vllm import LLM, SamplingParams
from peft import LoraConfig, get_peft_model
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
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


def sample_rollout_filter(
    model: LLM, data: list[dict[str, str]], sampling_param: SamplingParams
) -> list[dict[str, str]]:
    prompt_file = os.path.join(os.path.dirname(__file__), "prompts", "r1_zero.prompt")

    with open(prompt_file, "r") as f:
        prompt_template = f.read()

    prompt_list = [
        prompt_template.format(question=sample["question"]) for sample in data
    ]

    responses = model.generate(prompt_list, sampling_param)
    filtered_data = []
    for sample, response in zip(data, responses):
        for output in response.outputs:
            reward_dict = r1_zero_reward_fn(output.text, sample["answer"])
            if reward_dict["reward"] > 0.5:
                filtered_data.append(
                    {
                        "prompt": prompt_template.format(question=sample["question"]),
                        "output": output.text,
                    }
                )
    return filtered_data


def train_script(
    model_name: str = "Qwen/Qwen2.5-Math-1.5B",
    expt_name: str = "Qwen2.5_Math_1.5B_EI",
    n_ei_step: int = 5,
    ei_batch_size: int = 2048,
    n_rollout: int = 10,
    micro_batch_size: int = 4,
    gradient_accumulation_steps: int = 4,
    n_sft_epoch: int = 1,
    lr: float = 1e-5,
    use_lora: bool = True,
):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("using device: %s", device)

    # train model; to be initialized later
    policy_model: torch.nn.Module | None = None
    tokenizer: PreTrainedTokenizerBase | None = None
    optimizer: torch.optim.Optimizer | None = None
    rollout_model_path: str = model_name

    # define the train dataset
    data = load_gsm8k_train_data()
    assert ei_batch_size <= len(data)
    ei_loader = DataLoader(
        data,  # type: ignore
        batch_size=ei_batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=lambda batch: batch,
    )

    ei_step_idx = 0
    while ei_step_idx < n_ei_step:
        for ei_batch in ei_loader:
            # load the rollout model from local path
            logger.info("ei step: %d", ei_step_idx)
            vllm = LLM(rollout_model_path)
            sampling_param = get_rollout_samping_param(n_rollout)
            ei_sampled_data = sample_rollout_filter(vllm, ei_batch, sampling_param)
            logger.info(
                "sampled %d ei outputs from %d questions",
                len(ei_sampled_data),
                len(ei_batch),
            )
            if len(ei_sampled_data) == 0:
                logger.info("retry ei sampling")
                continue
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
                    policy_model = get_peft_model(policy_model, lora_config)
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
            # construct the training data for the current ei step
            sft_loader = DataLoader(
                ei_sampled_data,  # type: ignore
                batch_size=micro_batch_size,
                shuffle=True,
                collate_fn=lambda batch: tokenize_prompt_and_output(
                    [s["prompt"] for s in batch],
                    [s["output"] for s in batch],
                    tokenizer,  # type: ignore
                ),
            )
            sft_training_loop(
                policy_model,
                sft_loader,
                optimizer,
                device,
                gradient_accumulation_steps,
                n_sft_epoch,
            )
            rollout_model_path = os.path.join(
                os.path.dirname(__file__),
                "..",
                "train_output",
                expt_name,
                f"ei_step_{ei_step_idx}",
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

            ei_step_idx += 1


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
        "--expt", type=str, default="Qwen2.5_Math_1.5B_EI", dest="expt_name"
    )
    parser.add_argument("--ei_steps", type=int, default=5, dest="n_ei_step")
    parser.add_argument("--ei_batch_size", type=int, default=2048)
    parser.add_argument("--n_rollout", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4, dest="micro_batch_size")
    parser.add_argument(
        "--accum_steps", type=int, default=4, dest="gradient_accumulation_steps"
    )
    parser.add_argument("--sft_epochs", type=int, default=1, dest="n_sft_epoch")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument(
        "--use_lora", action=argparse.BooleanOptionalAction, default=True
    )
    args = parser.parse_args()
    train_script(
        model_name=args.model_name,
        expt_name=args.expt_name,
        n_ei_step=args.n_ei_step,
        ei_batch_size=args.ei_batch_size,
        n_rollout=args.n_rollout,
        micro_batch_size=args.micro_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        n_sft_epoch=args.n_sft_epoch,
        lr=args.lr,
        use_lora=args.use_lora,
    )
