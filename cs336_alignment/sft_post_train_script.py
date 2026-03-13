from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import json
import torch
from cs336_alignment.util import (
    tokenize_prompt_and_output,
    get_response_log_probs,
    sft_microbatch_train_step,
)
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
import logging
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from cs336_alignment.gsm_benchmark_script import evaluate_gsm
from vllm import LLM, SamplingParams
from peft import LoraConfig, get_peft_model


logger = logging.getLogger(__name__)


def load_gsm8k_train_data() -> list[dict]:
    # load prompt template
    prompt_file = os.path.join(os.path.dirname(__file__), "prompts", "r1_zero.prompt")
    with open(prompt_file, "r") as f:
        prompt_template = f.read()
    # load the training data from the jsonl file
    train_data_path = os.path.join(
        os.path.dirname(__file__), "..", "data", "gsm8k", "train.jsonl"
    )
    with open(train_data_path, "r") as f:
        train_data = [json.loads(line) for line in f]
    train_data_processed = []
    # preprocess train data
    for train_sample in train_data:
        question = train_sample["question"]
        answer = train_sample["answer"]
        prompt = prompt_template.format(question=question)
        # <think> token is already included in the prompt
        idx = answer.rfind("####")
        output = (
            answer[:idx] + "</think> <answer>" + answer[idx + 4 :].strip() + "</answer>"
        )
        # sanity check
        if r1_zero_reward_fn(output, answer)["reward"] > 0.5:
            train_data_processed.append(
                {
                    "question": question,
                    "answer": answer,
                    "prompt": prompt,
                    "output": output
                }
            )
    logger.info(
        "loaded %d / %d train samples",
        len(train_data_processed),
        len(train_data)
    )

    return train_data_processed


def sft_training_loop(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    gradient_accumulation_steps: int,
    n_epoch: int,
) -> None:
    step_idx = 0
    update_idx = 0
    optimizer.zero_grad()
    for epoch_idx in range(n_epoch):
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            response_mask = batch["response_mask"].to(device)
            log_probs = get_response_log_probs(model, input_ids, labels, True)
            loss, step_data = sft_microbatch_train_step(
                log_probs["log_probs"], response_mask, gradient_accumulation_steps, 1.0  # type: ignore
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
                        log_probs["token_entropy"].mean().item()
                    )
                update_idx += 1
            step_idx += 1


def train_script(
    model_name: str = "Qwen/Qwen2.5-Math-1.5B",
    expt_name: str = "Qwen2.5_Math_1.5B_SFT",
    micro_batch_size: int = 1,
    gradient_accumulation_steps: int = 16,
    n_epoch: int = 1,
    lr: float = 1e-5,
    use_lora: bool = True,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("using device: %s", device)

    # load the model training
    model = AutoModelForCausalLM.from_pretrained(
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
        model = get_peft_model(model, lora_config)
    model.to(device)

    # define the train dataset
    data = load_gsm8k_train_data()
    loader = DataLoader(
        data,  # type: ignore
        batch_size=micro_batch_size,
        shuffle=True,
        collate_fn=lambda batch: tokenize_prompt_and_output(
            [s["prompt"] for s in batch],
            [s["output"] for s in batch],
            tokenizer,
        ),
    )

    # optimizer
    optimizer = AdamW(model.parameters(), lr=lr)

    # sampling setting for eval
    param = SamplingParams(temperature=1.0, top_p=1.0, max_tokens=1024)
    param.stop = ["</answer>"]
    param.include_stop_str_in_output = True

    sft_training_loop(
        model, loader, optimizer, device, gradient_accumulation_steps, n_epoch
    )

    # save the trained model + tokenizer
    output_dir = os.path.join(
        os.path.dirname(__file__), "..", "train_output", expt_name
    )
    os.makedirs(output_dir, exist_ok=True)
    if use_lora:
        model = model.merge_and_unload()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # load the trained model for eval
    model.cpu()
    del model
    del optimizer
    torch.cuda.empty_cache()
    model = LLM(output_dir)

    evaluate_gsm(expt_name, model, param)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(filename)s:%(lineno)d %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    train_script()
