from typing import Optional, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from src.models.modules import RMSNorm, TransformerBlock, precompute_freqs_cis


class KoreanLLM(nn.Module):
    def __init__(
        self,
        vocab_size: int = 128256,
        pad_token_id: int = 128004,
        dim: int = 1920,
        n_layers: int = 20,
        n_heads: int = 10,
        max_seq_len: int = 2048
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads

        self.embed = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList([
            TransformerBlock(dim, n_heads, int(dim * 2.5))
            for _ in range(n_layers)
        ])
        self.norm = RMSNorm(dim)
        self.output = nn.Linear(dim, vocab_size, bias=False)
        self.output.weight = self.embed.weight

        f_cos, f_sin = precompute_freqs_cis(self.head_dim, max_seq_len * 2)
        self.register_buffer("f_cos", f_cos)
        self.register_buffer("f_sin", f_sin)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)

    def _get_freqs(self, f: torch.Tensor, start: int, length: int) -> torch.Tensor:
        return f[start:start + length]

    def forward(
        self,
        tokens: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        kv_caches: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], List[Tuple[torch.Tensor, torch.Tensor]]]:
        b, s = tokens.shape
        x = self.embed(tokens)

        start_pos = 0
        if kv_caches is not None and len(kv_caches) > 0 and kv_caches[0][0] is not None:
            start_pos = kv_caches[0][0].shape[2]

        f_cos = self._get_freqs(self.f_cos, start_pos, s)
        f_sin = self._get_freqs(self.f_sin, start_pos, s)

        new_kv_caches = []
        for i, layer in enumerate(self.layers):
            if self.training:
                x, kv = checkpoint(
                    layer, x, f_cos, f_sin, None,
                    use_reentrant=False
                )
            else:
                kv_cache = kv_caches[i] if kv_caches else None
                x, kv = layer(x, f_cos, f_sin, kv_cache=kv_cache)

            new_kv_caches.append(kv)

        x = self.norm(x)
        logits = self.output(x)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits[..., :-1, :].reshape(-1, logits.size(-1)),
                labels[..., 1:].reshape(-1),
                ignore_index=self.pad_token_id,
                reduction='mean'
            )

        return logits, loss, new_kv_caches
