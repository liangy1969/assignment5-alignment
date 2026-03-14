from math import e
import os

from vllm import LLM, SamplingParams
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
import json
from typing import Callable


def main_baseline():
    exp_name = "Qwen2.5_Math_1.5B_GSM_baseline"
    model = LLM("Qwen/Qwen2.5-Math-1.5B")
    param = SamplingParams(temperature=1.0, top_p=1.0, max_tokens=1024)
    param.stop = ["</answer>"]
    param.include_stop_str_in_output = True
    evaluate_gsm(exp_name, model, param)


def evaluate_gsm(exp_name: str, model: LLM, sampling_params: SamplingParams) -> None:

    prompt_file = os.path.join(os.path.dirname(__file__), "prompts", "r1_zero.prompt")

    with open(prompt_file, "r") as f:
        prompt_template = f.read()

    test_data_path = os.path.join(
        os.path.dirname(__file__), "..", "data", "gsm8k", "test.jsonl"
    )
    with open(test_data_path, "r") as f:
        test_data = [json.loads(line) for line in f]

    test_prompt_list = [
        prompt_template.format(question=sample["question"], answer=sample["answer"])
        for sample in test_data
    ]

    responses = model.generate(test_prompt_list, sampling_params)
    records = []
    for data, response in zip(test_data, responses):
        q = data["question"]
        gt_ans = data["answer"]
        model_ans = response.outputs[0].text
        reward_dict = r1_zero_reward_fn(model_ans, gt_ans)
        records.append(
            {
                "question": q,
                "ground_truth": gt_ans,
                "model_answer": model_ans,
                "format_reward": reward_dict["format_reward"],
                "answer_reward": reward_dict["answer_reward"],
                "reward": reward_dict["reward"],
            }
        )

    output_path = os.path.join(
        os.path.dirname(__file__), "..", "exp_results", f"{exp_name}.jsonl"
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")

    avg_format_reward = sum(r["format_reward"] for r in records) / len(records)
    avg_reward = sum(r["reward"] for r in records) / len(records)
    print(f"Average format reward: {avg_format_reward:.4f}")
    print(f"Average reward: {avg_reward:.4f}")
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main_baseline()
