# 🏛️ Model Architecture & Optimization Guide

Korean LLM Advanced v3 is a **1.09B parameter Decoder-Only Transformer language model** built from scratch and tailored for Korean NLP.
This document details the underlying mathematical components, tensor specifications, and memory engineering techniques that reduced training VRAM from 23GB to 9GB.

---

## 1. Model Specifications

| Hyperparameter | Value | Description |
| :--- | :--- | :--- |
| **Total Parameters** | ~1.09B (1,093M) | Sum of embedding and all transformer layer weights |
| **Hidden Dimension (`dim`)** | 1,920 | Input dimension for token embeddings and attention layers |
| **Number of Layers (`n_layers`)** | 20 | Number of stacked transformer blocks |
| **Attention Heads (`n_heads`)** | 10 | Number of multi-head attention heads |
| **Dimension per Head (`head_dim`)** | 192 (`1920 // 10`) | Projection dimension per attention head |
| **FFN Hidden Dimension (`hidden_dim`)** | 4,800 (`int(dim * 2.5)`) | SwiGLU feedforward expansion dimension |
| **Max Context Length (`max_seq_len`)** | 2,048 Tokens | Precomputed RoPE frequency buffer limit |
| **Vocabulary Size (`vocab_size`)** | 128,256+ | Based on `beomi/Llama-3-Open-Ko-8B` tokenizer (includes `<|pad|>`) |

---

## 2. Core Architectural Components

### 1) RMSNorm (Root Mean Square Normalization)
Omits the mean-centering step found in conventional LayerNorm, scaling purely by root mean square variance to boost execution speed by 10~15%:
$$\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d} \sum_{i=1}^{d} x_i^2 + \epsilon}} \odot \gamma$$

### 2) RoPE (Rotary Position Embedding)
Applies rotation matrices to Query and Key projections rather than adding absolute positional vectors. This embeds relative distance information directly into the attention dot products, maintaining positional fidelity across long contexts.

### 3) SwiGLU Feed-Forward Network
Combines Gated Linear Units with SiLU activation functions, delivering richer expressive capacity compared to conventional standard ReLU/GELU:
$$\text{SwiGLU}(x) = W_2 (\text{SiLU}(W_1 x) \odot W_3 x)$$

### 4) SDPA (Scaled Dot-Product Attention) & KV Cache
- **During Training**: Utilizes PyTorch's native `F.scaled_dot_product_attention` for kernel-level fused memory access and causal masking.
- **During Inference**: Retains past Key and Value states in a `kv_cache` buffer, bypassing full-sequence recomputation for $O(N)$ autoregressive generation.

---

## 3. VRAM Optimization Engineering (Achieving ~9GB Footprint)

```
[Memory Optimization Pipeline]
Standard FP32 Training (~23GB)
  ↓ BF16 Mixed Precision (-50% model weights & activation memory)
  ↓ 8-bit AdamW Optimizer (-75% optimizer state memory)
  ↓ Gradient Checkpointing (-35% activation memory)
Final VRAM Footprint: ~9GB (Trainable on single consumer GPU)
```

1. **BF16 Automatic Mixed Precision (Bfloat16 AMP)**:
   - Matches FP32 exponent dynamic range (8 bits), eliminating underflow risks without gradient scalers while cutting memory in half.
2. **8-bit AdamW Optimizer (`bitsandbytes`)**:
   - Compresses 1st and 2nd optimizer momentums using block-wise non-linear quantization.
3. **Gradient Checkpointing (`torch.utils.checkpoint`)**:
   - Recomputes intermediate layer activations on the backward pass rather than caching everything during forward propagation.
4. **Gradient Accumulation**:
   - Combines micro-batches (`batch_size=2`) with `accumulation_steps=8`~`32` to simulate effective batch sizes of 16~64 within physical hardware limits.
