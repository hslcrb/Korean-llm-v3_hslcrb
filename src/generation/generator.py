from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


@torch.no_grad()
def generate(
    model: nn.Module,
    tokenizer,
    prompt: str = "안녕? 너는 누구니?",
    max_tokens: int = 512,
    temperature: float = 0.6,
    top_k: int = 40,
    top_p: float = 0.95,
    repetition_penalty: float = 1.3,
    device: Optional[torch.device] = None
) -> str:
    if device is None:
        device = next(model.parameters()).device

    was_training = model.training
    model.eval()

    prompt_text = f"### 질문: {prompt}\n### 응답:"
    tokens = tokenizer.encode(prompt_text, return_tensors="pt").to(device)

    kv_caches = None
    output_tokens = tokens

    for step in range(max_tokens):
        input_tokens = output_tokens[:, -1:] if kv_caches is not None else output_tokens

        with torch.no_grad():
            logits, _, kv_caches = model(input_tokens, kv_caches=kv_caches)

        next_logits = logits[:, -1, :] / temperature

        if repetition_penalty != 1.0:
            for token_id in set(output_tokens[0].tolist()):
                if next_logits[0, token_id] < 0:
                    next_logits[0, token_id] *= repetition_penalty
                else:
                    next_logits[0, token_id] /= repetition_penalty

        if top_k > 0:
            indices_to_remove = next_logits < torch.topk(next_logits, min(top_k, next_logits.size(-1)))[0][..., -1, None]
            next_logits[indices_to_remove] = float('-inf')

        probs = F.softmax(next_logits, dim=-1)

        if top_p < 1.0:
            sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
            cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
            sorted_indices_to_remove = cumsum_probs > top_p
            sorted_indices_to_remove[..., 0] = False
            indices_to_remove = torch.zeros_like(probs, dtype=torch.bool)
            indices_to_remove.scatter_(dim=-1, index=sorted_indices, src=sorted_indices_to_remove)
            probs[indices_to_remove] = 0.0
            probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-10)

        next_token = torch.multinomial(probs, num_samples=1)
        output_tokens = torch.cat([output_tokens, next_token], dim=1)

        if next_token.item() == tokenizer.eos_token_id:
            break

        if output_tokens.shape[1] > 512:
            break

    generated_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
    response = generated_text.split("### 응답:")[-1].strip() if "### 응답:" in generated_text else generated_text

    if was_training:
        model.train()
    return response
