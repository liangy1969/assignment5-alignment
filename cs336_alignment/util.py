from __future__ import annotations


import torch
from torch import Tensor
from transformers import PreTrainedTokenizerBase
from einops import rearrange, reduce
from typing import Callable, Optional


def tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer: PreTrainedTokenizerBase,
) -> dict[str, Tensor]:
    """Tokenize the prompt and output strings, and construct a mask that is 1
    for the response tokens and 0 for other tokens (prompt or padding).

    Args:
        prompt_strs: list[str], the prompt strings.
        output_strs: list[str], the output strings.
        tokenizer: PreTrainedTokenizer, the tokenizer to use.

    Returns:
        dict[str, torch.Tensor]:
            "input_ids": torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1):
                the tokenized prompt and output strings, with the final token sliced off.
            "labels": torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1):
                shifted input_ids (i.e., the input_ids without the first token).
            "response_mask": torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1):
                a mask on the response tokens in `labels`.
    """
    prompt_lens = [len(tokenizer.encode(p)) for p in prompt_strs]
    output_lens = [len(tokenizer.encode(o)) for o in output_strs]
    prompt_and_output_len = max(pl + ol for pl, ol in zip(prompt_lens, output_lens)) - 1
    prompt_tokens = tokenizer(
        prompt_strs,
        padding=True,
        truncation=False,
        return_tensors="pt",
    )
    output_tokens = tokenizer(
        output_strs,
        padding=True,
        truncation=False,
        return_tensors="pt",
    )
    input_ids_list: list[Tensor] = []
    labels_list: list[Tensor] = []
    response_mask_list: list[Tensor] = []
    for i in range(len(prompt_strs)):
        # Shift the prompt length by 1 to account for the final token being sliced off.
        prompt_token = prompt_tokens["input_ids"][i]
        output_token = output_tokens["input_ids"][i]
        output_mask = output_tokens["attention_mask"][i]
        prompt_len = prompt_lens[i]
        output_len = output_lens[i]
        total_len = prompt_len + output_len - 1
        pad_len = prompt_and_output_len - total_len
        input_ids = torch.cat(
            [prompt_token[:prompt_len], output_token[:output_len]], dim=0
        )
        label_ids = torch.cat(
            [prompt_token[1:prompt_len], output_token[:output_len]], dim=0
        )
        input_ids_padded = torch.cat(
            [
                input_ids,
                torch.full(
                    (pad_len,),
                    tokenizer.pad_token_id,
                    dtype=input_ids.dtype,
                    device=input_ids.device,
                ),
            ],
            dim=0,
        )[:prompt_and_output_len]
        labels_padded = torch.cat(
            [
                label_ids,
                torch.full(
                    (pad_len,),
                    tokenizer.pad_token_id,
                    dtype=label_ids.dtype,
                    device=label_ids.device,
                ),
            ],
            dim=0,
        )
        response_mask = torch.cat(
            [
                torch.zeros(
                    prompt_len - 1, dtype=output_mask.dtype, device=output_mask.device
                ),
                output_mask[:output_len],
                torch.zeros(
                    pad_len, dtype=output_mask.dtype, device=output_mask.device
                ),
            ],
            dim=0,
        )
        input_ids_list.append(input_ids_padded)
        labels_list.append(labels_padded)
        response_mask_list.append(response_mask)

    return {
        "input_ids": torch.stack(input_ids_list, dim=0),
        "labels": torch.stack(labels_list, dim=0),
        "response_mask": torch.stack(response_mask_list, dim=0),
    }


def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Get the entropy of the logits (i.e., entropy of the final dimension)."""
    # formulata: (softmax)
    # log(sum_i(exp(x[i]))) - (sum_i(x[i] * exp(x[i]))) / (sum_i(exp(x[i])))
    log_sum_exp = torch.logsumexp(logits, dim=-1)
    max_logits = torch.max(logits, dim=-1).values
    logit_minus_max = logits - max_logits.unsqueeze(-1)
    exp_logits = torch.exp(logit_minus_max)
    return log_sum_exp - torch.sum(logits * exp_logits, dim=-1) / torch.sum(
        exp_logits, dim=-1
    )


def get_response_log_probs(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool,
) -> dict[str, torch.Tensor | None]:
    """Get the conditional log-probs of the response given the prompt,
        and optionally the entropy of the next token predictions.

    Args:
        model: PreTrainedModel, the model to score.
        input_ids: torch.Tensor of shape (batch_size, sequence_length):
            the tokenized prompt and output.
        labels: torch.Tensor of shape (batch_size, sequence_length):
            shifted input_ids.
        return_token_entropy: bool, whether to return the entropy of the
            next token predictions.

    Returns:
        dict[str, torch.Tensor | None]:
            "log_probs": torch.Tensor of shape (batch_size, sequence_length):
                the conditional log-probs of the response given the prompt.
                Note that we have not masked out the token indices corresponding
                to the prompt or padding; that is done in the train loop.
            "token_entropy": Optional[torch.Tensor] of shape (batch_size, sequence_length):
                the entropy of the next token predictions. As with the log-probs,
                we have not masked out the token indices corresponding to the prompt
                or padding; that is done in the train loop.
    """
    logits = model(input_ids).logits
    log_probs_all = torch.nn.functional.log_softmax(logits, dim=-1)
    log_probs = torch.gather(log_probs_all, dim=-1, index=labels.unsqueeze(-1)).squeeze(
        -1
    )
    token_entropy = compute_entropy(logits) if return_token_entropy else None
    return {"log_probs": log_probs, "token_entropy": token_entropy}


def masked_normalize(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: int | None = None,
    normalize_constant: float = 1.0,
) -> torch.Tensor:
    """Sum over a dimension and normalize by a constant,
    considering only the elements with mask value 1.

    Args:
        tensor: torch.Tensor, the tensor to sum and normalize.
        mask: torch.Tensor, the mask. We only consider elements
            with mask value 1.
        dim: int | None, the dimension to sum along before
            normalization. If None, sum over all dimensions.
        normalize_constant: float, the constant to divide by
            for normalization.

    Returns:
        torch.Tensor, the normalized sum, where masked elements
            (mask=0) don't contribute to the sum.
    """
    masked_tensor = tensor * mask
    sum_masked_tensor = masked_tensor.sum(dim=dim)
    normalized_sum = sum_masked_tensor / normalize_constant
    return normalized_sum


def sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: float | None = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the policy gradient loss and backprop its gradients for a microbatch."""
    if normalize_constant is None:
        normalize_constant = 1.0
    masked_log_prob_sum = masked_normalize(
        policy_log_probs,
        response_mask,
        -1,
        normalize_constant * gradient_accumulation_steps,
    )
    loss = -masked_log_prob_sum.mean()
    loss.backward()
    return loss, {"masked_log_prob_sum": masked_log_prob_sum}


def compute_group_normalized_rewards(
    reward_fn: Callable,
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    advantage_eps: float,
    normalize_by_std: bool,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """
    Compute rewards for each group of rollout responses,
    normalized by the group size.

    For more on GRPO, see:
        DeepSeekMath: https://arxiv.org/abs/2402.03300
        DeepSeek-R1: https://arxiv.org/abs/2501.12948

    Args:
        reward_fn: Callable[[str, str], dict[str, float]],
            scores the rollout responses against the ground truths,
            producing a dict with keys
            "reward", "format_reward", and "answer_reward".
        rollout_responses: list[str], rollouts from the policy.
            The length of this list is
            `rollout_batch_size = n_prompts_per_rollout_batch * group_size`.
        repeated_ground_truths: list[str], the ground truths for the examples.
            The length of this list is `rollout_batch_size`,
            because the ground truth for each example is repeated `group_size` times.
        group_size: int, number of rollouts per group.
        advantage_eps: float, epsilon to avoid division by zero
            during group normalization.
        normalize_by_std: bool, whether to normalize the rewards by
            std(rewards).

    Returns:
        tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
            torch.Tensor of shape (rollout_batch_size,):
                group-normalized rewards for each rollout response.
            torch.Tensor of shape (rollout_batch_size,):
                raw rewards for each rollout response.
            dict[str, float]: metadata for the rewards of the rollout batch.
                You may choose what you wish to log here
                (some statistics of the rewards, etc.).
    """
    reward_meta_list = []
    for response, gt in zip(rollout_responses, repeated_ground_truths):
        reward_meta_list.append(reward_fn(response, gt))

    reward_tensor = torch.tensor(
        [reward_meta["reward"] for reward_meta in reward_meta_list]
    )
    reward_tensor = rearrange(reward_tensor, "(b g) -> b g", g=group_size)
    reward_mean = reduce(reward_tensor, "b g -> b", "mean")
    reward_tensor_norm = reward_tensor - rearrange(reward_mean, "b -> b 1")
    if normalize_by_std:
        reward_std = reward_tensor.std(dim=-1)
        reward_tensor_norm /= rearrange(reward_std, "b -> b 1") + advantage_eps
    reward_tensor_norm = rearrange(reward_tensor_norm, "b g -> (b g)")
    reward_tensor = rearrange(reward_tensor, "b g -> (b g)")

    return reward_tensor_norm, reward_tensor, {}
