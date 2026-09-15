# 🇰🇷 Korean LLM Advanced v3

<div align="center">

[![License: GPL-3.0](https://img.shields.io/badge/License-GPL%203.0-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-brightgreen)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C)](https://pytorch.org)
[![CUDA](https://img.shields.io/badge/CUDA-11.8%2B-76B900)](https://developer.nvidia.com/cuda-toolkit)
[![Model Size](https://img.shields.io/badge/Model-1.09B%20Parameters-orange)](#-specifications)
[![VRAM](https://img.shields.io/badge/VRAM%20Usage-~9GB-red)](#-vram-comparison)

**A 1.09B Parameter Lightweight Korean LLM Built from Scratch with Advanced VRAM Optimization**

[한국어](README.md) • [📖 Development Story](docs/DEVELOPMENT_STORY_EN.md) • [🏛️ Architecture Guide](docs/ARCHITECTURE_EN.md) • [🚀 Quick Start](#-quick-start) • [❓ FAQ](#-frequently-asked-questions-faq)

</div>

---

## 📖 Overview

**Korean LLM Advanced v3** is a **1.09-billion parameter lightweight Korean language model** designed for efficient pretraining and fine-tuning on consumer-grade GPUs and local development environments.

Originally created from scratch by an 8th-grade middle school developer—advancing from a 50M prototype and 541M v1 to the full 1.09B v2/v3 architecture—this open-source project incorporates cutting-edge memory optimization techniques to **reduce VRAM consumption from 23GB down to ~9GB**, making training viable on single consumer graphics cards.

> 💡 **Behind the Scenes**: Read the detailed story of 44,000 training steps, debugging instruction-following failures, and mastering quantization in our [**Development Story (docs/DEVELOPMENT_STORY_EN.md)**](docs/DEVELOPMENT_STORY_EN.md).

---

## 🌟 Key Features

### 🎯 Specifications
| Parameter | Value | Remarks |
| :--- | :--- | :--- |
| **Model Size** | **1.09B** (~1,093M parameters) | Embeddings + 20 Transformer Layers |
| **Hidden Dimension (`dim`)** | 1,920 | 192 per attention head |
| **Layers (`n_layers`)** | 20 | Stacked Transformer Blocks |
| **Attention Heads (`n_heads`)** | 10 | Multi-Head Attention |
| **FFN Dimension (`hidden_dim`)** | 4,800 | SwiGLU Gated Feedforward |
| **Max Context Length** | 2,048 Tokens | Precomputed RoPE Frequency Buffer |
| **Tokenizer Vocabulary** | 128,256+ | `beomi/Llama-3-Open-Ko-8B` (includes `<|pad|>`) |

---

### 🔧 5 Core Memory Optimizations

1. **BF16 Automatic Mixed Precision (Bfloat16 AMP)**
   - Halves parameter and activation memory compared to FP32 (**12GB → 6GB**).
   - Retains FP32-equivalent dynamic exponent range to ensure numeric stability without gradient scaling.
2. **8-bit AdamW Optimizer (`bitsandbytes`)**
   - Cuts optimizer state memory by **75%** (~2.2GB standard down to ~0.55GB).
3. **Gradient Checkpointing (`torch.utils.checkpoint`)**
   - Recomputes activations during the backward pass instead of caching them, saving 30~40% activation VRAM.
4. **Gradient Accumulation**
   - Pairs micro-batch size 2 with 8~32 accumulation steps to achieve effective batch sizes of 16~64 with minimal memory footprint.
5. **Dynamic Quantization & Fast Attention (SDPA + KV Cache)**
   - Accelerated via PyTorch's native `F.scaled_dot_product_attention` kernel with fast KV Cache autoregressive decoding.

---

## 💾 VRAM Comparison

<div align="center">

| Version | Model Scale | Typical VRAM | Optimization Applied |
| :---: | :---: | :---: | :--- |
| **v1** | 541M | ~11 GB | Standard FP32 |
| **v2** | ~1.1B | ~23 GB | BF16 + Gradient Checkpointing |
| **v3 (Current)** ⭐ | **1.09B** | **~9 GB** ✨ | **BF16 + 8-bit AdamW + Checkpointing + Quantization** |

*v3 slashes VRAM usage by over 60% compared to v2 while preserving the full 1B architecture.*

</div>

---

## 🚀 Quick Start

### 1. Prerequisites
- **Python**: 3.9+ (3.11 / 3.12 recommended)
- **GPU**: NVIDIA GPU with 9GB+ VRAM recommended (CPU inference supported)
- **CUDA**: 11.8 or newer

### 2. Installation

```bash
# 1. Clone repository
git clone https://github.com/seoan1024/korean-llm-v3.git
cd korean-llm-v3

# 2. Install all required dependencies
pip install -r requirements.txt
```

### 3. Training Execution

```bash
# Run training using the unified main entrypoint
python korean_llm_advanced_v3.py

# Custom training configuration with CLI arguments
python korean_llm_advanced_v3.py --batch-size 2 --accumulation-steps 8 --learning-rate 5e-5 --max-steps 50000
```

> 💡 **Automated Datasets**: Automatically downloads `nlpai-lab/kullm-v2` and `beomi/KoAlpaca-v1.1a` datasets and caches them securely in local Parquet format.

### 4. Real-time GUI Monitoring
Starting training automatically launches an integrated Tkinter & Matplotlib monitor:
- 📉 **Real-time Loss Curve**: Tracks training loss trends dynamically.
- 💬 **Interactive Chat Window**: Loads the latest checkpoint on CPU in the background for live conversational generation tests.

---

## 🏗️ Project Structure

```
korean-llm-v3/
├── korean_llm_advanced_v3.py         # 🚀 [Main] Unified CLI entrypoint and backward-compatible core
├── requirements.txt                   # Dependency specification
├── .gitignore                         # Git tracking exclusions
├── LICENSE                            # GNU GPL-3.0 License
│
├── src/                               # 📦 Modular Source Code
│   ├── __init__.py                    # __version__ = "v3.1.0" & top-level symbol exports
│   ├── config.py                      # TrainingConfig and global path constants
│   ├── models/                        # Transformer architecture (KoreanLLM, RMSNorm, RoPE)
│   ├── data/                          # DatasetManager, LocalKoreanDataset, collate_fn
│   ├── training/                      # Training loop and distributed setup
│   ├── generation/                    # KV Cache text generation engine
│   ├── gui/                           # TrainingMonitorGUI application
│   └── utils/                         # Checkpointing and logging utilities
│
├── docs/                              # 📚 Documentation
│   ├── DEVELOPMENT_STORY.md           # [Korean] Developer journey and reflections
│   ├── DEVELOPMENT_STORY_EN.md        # [English] Project Development Journey
│   ├── ARCHITECTURE.md                # [Korean] In-depth Architecture & Optimization Guide
│   └── ARCHITECTURE_EN.md             # [English] Model Architecture & Optimization Guide
│
├── archive/                           # 🗄️ Legacy Script Archive
│   ├── README.md                      # Archive notice (historical preservation)
│   └── korean_llm_advanced_v3tresure.py # Pre-refactoring monolithic script
│
├── checkpoints/                       # 💾 Saved model checkpoints (.pth)
├── datasets/cache/                    # 📥 Cached Parquet datasets
└── logs/                              # 📝 Logs and loss_history.json
```

---

## 🔧 Technology Stack

| Library | Version | Purpose |
| :--- | :--- | :--- |
| **PyTorch** | 2.0+ | Deep learning framework & SDPA computation |
| **Transformers** | 4.30+ | `Llama-3-Open-Ko-8B` tokenizer & LR scheduler |
| **Datasets** | 2.10+ | Hugging Face Korean instruction dataset loader |
| **bitsandbytes** | 0.43+ | 8-bit AdamW optimizer & quantization |
| **pyarrow** | 12.0+ | High-speed Parquet data processing |
| **matplotlib** | 3.7+ | Real-time loss visualization |
| **tkinter** | Built-in | GUI monitoring and chat interface |
| **tqdm / pandas** | Latest | Progress tracking and dataframe utilities |

---

## 💡 Code Examples

### 1. Checkpoint Inference
```python
import torch
from transformers import AutoTokenizer
from src.models import KoreanLLM
from src.generation import generate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("beomi/Llama-3-Open-Ko-8B")
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

model = KoreanLLM(vocab_size=len(tokenizer), pad_token_id=tokenizer.pad_token_id).to(device)

checkpoint = torch.load("checkpoints/korean_llm_50000.pth", map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

response = generate(model, tokenizer, prompt="인공지능이란 무엇인가요?", max_tokens=128, device=device)
print("Response:", response)
```

### 2. Custom Training Config
```python
from src.config import TrainingConfig
from src.training import train

config = TrainingConfig(
    batch_size=2,
    accumulation_steps=16,
    max_steps=50000,
    learning_rate=3e-5,
    warmup_steps=300,
    use_bfloat16=True,
    resume_from_checkpoint="latest"
)

train(config)
```

---

## ❓ Frequently Asked Questions (FAQ)

<details>
<summary><b>Q1. How can I run inference without full training?</b></summary>
<p>
Load any saved checkpoint file (.pth) into <code>KoreanLLM</code> as shown in the code example above and call <code>generate()</code> for fast text generation.
</p>
</details>

<details>
<summary><b>Q2. My GPU has less than 8GB VRAM. How should I configure it?</b></summary>
<p>
Adjust the configuration as follows:
1. Set <code>batch_size=1</code>
2. Shorten <code>max_seq_len</code> to <code>256</code> or <code>512</code>
3. Increase <code>accumulation_steps=16</code> or higher to maintain effective batch sizing.
</p>
</details>

<details>
<summary><b>Q3. How do I resume interrupted training?</b></summary>
<p>
Run <code>python korean_llm_advanced_v3.py --resume latest</code>. It automatically locates the latest step in <code>checkpoints/</code> and restores model weights, optimizer states, and scheduler steps.
</p>
</details>

<details>
<summary><b>Q4. Can I integrate other custom datasets?</b></summary>
<p>
Yes. Add dataset names, splits, and text keys to <code>DATASETS_CONFIG</code> in <code>src/data/dataset_manager.py</code>. The script automatically converts and caches them as Parquet.
</p>
</details>

<details>
<summary><b>Q5. DataLoader multi-worker error on Windows?</b></summary>
<p>
Set <code>TrainingConfig(num_workers=0)</code> to avoid Windows process-spawning collisions.
</p>
</details>

<details>
<summary><b>Q6. Can this run on CPU without an NVIDIA GPU?</b></summary>
<p>
Yes. The codebase auto-detects CUDA availability and falls back smoothly to CPU mode. (Note: CPU training for a 1B model is slow; CPU mode is primarily recommended for testing and evaluation.)
</p>
</details>

<details>
<summary><b>Q7. What is the archive folder for?</b></summary>
<p>
The <code>archive/</code> directory safely preserves the pre-refactoring original single-file script for historical integrity. It is not used in current training workflows.
</p>
</details>

<details>
<summary><b>Q8. Can I export the model to ONNX?</b></summary>
<p>
Yes. Use standard <code>torch.onnx.export</code> with a dummy token tensor to serialize <code>KoreanLLM</code> into an ONNX graph.
</p>
</details>

<details>
<summary><b>Q9. Can I use this for commercial or derivative projects?</b></summary>
<p>
This project is licensed under the <b>GNU General Public License v3.0 (GPL-3.0)</b>. Derivative open-source projects are welcomed under GPL-3.0 compliance.
</p>
</details>

---

## 📜 License & Contributions

- **License**: [GNU General Public License v3.0 (GPL-3.0)](./LICENSE)
- **Developer**: [seoan1024](https://github.com/seoan1024) (Contact: seoan102410@gmail.com)
- Pull requests, discussions, and feature suggestions are warmly welcomed!

<div align="center">

**⭐ If you find this project inspiring, please leave a Star on GitHub! ⭐**

</div>
