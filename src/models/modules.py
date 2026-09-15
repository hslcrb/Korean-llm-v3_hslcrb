from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# 1. RMSNorm
# ==========================================
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


# ==========================================
# 2. RoPE (Rotary Position Embedding)
# ==========================================
def precompute_freqs_cis(head_dim: int, end: int, theta: float = 10000.0) -> Tuple[torch.Tensor, torch.Tensor]:
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2)[: (head_dim // 2)].float() / head_dim))
    t = torch.arange(end, dtype=freqs.dtype)
    freqs = torch.outer(t, freqs)
    return torch.cos(freqs), torch.sin(freqs)


def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    seq, head_dim_2 = cos.shape
    head_dim = head_dim_2 * 2

    x1, x2 = x[..., :head_dim // 2], x[..., head_dim // 2:]
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)

    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


# ==========================================
# 3. SwiGLU 피드포워드 네트워크
# ==========================================
class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


# ==========================================
# 4. Attention
# ==========================================
class Attention(nn.Module):
    def __init__(self, dim: int, n_heads: int):
        super().__init__()
        assert dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = dim // n_heads

        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, dim, bias=False)
        self.wv = nn.Linear(dim, dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        f_cos: torch.Tensor,
        f_sin: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        b, s, d = x.shape

        q = self.wq(x).view(b, s, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(b, s, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(b, s, self.n_heads, self.head_dim).transpose(1, 2)

        q = apply_rotary_emb(q, f_cos, f_sin)
        k = apply_rotary_emb(k, f_cos, f_sin)

        if kv_cache is not None:
            pk, pv = kv_cache
            k = torch.cat([pk, k], dim=2)
            v = torch.cat([pv, v], dim=2)

        new_kv = (k.detach(), v.detach())

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=mask,
            is_causal=(mask is None and s > 1)
        )

        out = out.transpose(1, 2).contiguous().view(b, s, d)
        return self.wo(out), new_kv


# ==========================================
# 5. TransformerBlock
# ==========================================
class TransformerBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, hidden_dim: int):
        super().__init__()
        self.attention = Attention(dim, n_heads)
        self.feed_forward = SwiGLU(dim, hidden_dim)
        self.attention_norm = RMSNorm(dim)
        self.ffn_norm = RMSNorm(dim)

    def forward(
        self,
        x: torch.Tensor,
        f_cos: torch.Tensor,
        f_sin: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        normed_x = self.attention_norm(x)
        h, new_kv = self.attention(normed_x, f_cos, f_sin, kv_cache=kv_cache)
        x = x + h
        x = x + self.feed_forward(self.ffn_norm(x))
        return x, new_kv
