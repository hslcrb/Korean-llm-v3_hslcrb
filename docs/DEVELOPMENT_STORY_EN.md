# 🚀 Project Development Journey

> **"When free API tiers tightened and the limits of 'vibe coding' hit, a question crossed my mind: 'Why not build an LLM from scratch with my own hands?'"**

---

## 💡 The Bold First Step

When I started building AI models, I was just an 8th-grade middle school student. The only thing I knew about deep learning was that the letter 'B' stands for 'Billion' parameters. Writing advanced neural network architectures felt like an impossible mountain to climb.

So, I opened ChatGPT and asked bluntly:  
**"I want to build my own independent Korean LLM from scratch!"**

At first, I spent days piecing together code snippets provided by ChatGPT, constantly praying that tensor dimension errors wouldn't crash the script. I spent entire summer vacation days glued to my screen, scraping, cleaning, and formatting Korean text datasets.

---

## 🐣 Trials, Errors, and Growth: From v1 to v2

### The 50M Prototype
My very first prototype was an ultra-lightweight 50M parameter model trained on Korean Wikipedia articles. Although the file no longer exists and it was incapable of holding a coherent conversation, seeing it piece together grammatically structured Korean phrases was exhilarating.

Motivated by that small win, I shifted my focus entirely toward building conversational instruction models, eventually scaling up to the **541M parameter** v1 model.

### 44,000 Steps: Triumph and Heartbreak (v2)
After resolving foundational bugs in v1, I doubled the model scale to **1.09B (1.09 billion) parameters**. My PC hummed continuously throughout the vacation, and training finally crossed the **44,000 step** milestone.

With trembling hands, I typed a simple greeting into the test prompt:

> **User:** "안녕?" (Hello?)  
> **Model:** "안녕하세요! 오늘은 무엇을 도와드릴까요?" (Hello! How can I assist you today?)

Seeing the model produce a clear, coherent Korean response on screen brought an unforgettable sense of accomplishment.

However, celebration was short-lived. As I tested more diverse prompts, the model began drifting off-topic, outputting erratic answers. After pulling an all-nighter debugging the architecture alongside AI, I discovered a critical flaw: the model was failing to attend properly to instruction formatting. It was heartbreaking, but to build a truly robust model, I made the tough decision to discard the 44,000-step checkpoint.

---

## ⚡ Conquering Hardware Constraints: From 23GB to 9GB VRAM (v3)

After rectifying the architectural issue, I faced another daunting bottleneck: training a 1.09B parameter transformer required **more than 23GB of VRAM**, far exceeding the capacity of standard consumer GPUs.

**"I must bring the memory footprint under 10GB so anyone can train and run it."**

To conquer this, I methodically researched and integrated modern memory optimization techniques:
- **BF16 Mixed Precision**: Halved model parameter memory consumption.
- **8-bit AdamW Optimizer (`bitsandbytes`)**: Reduced optimizer state memory by ~75%.
- **Gradient Checkpointing**: Drastically minimized activation memory at a modest computational cost.
- **Gradient Accumulation**: Allowed micro-batches (size=2) to simulate larger effective batch sizes without memory overhead.
- **Dynamic Quantization Support**: Optimized inference speed and memory.

Through these enhancements, v3 retained the full 1.09B architecture while slashing VRAM usage down to **approximately 9GB**.

---

## 🌟 Epilogue

With the new school semester underway, running sustained long-term pretraining on personal hardware has become difficult. Yet, this project stands as my proudest creation—built purely through stubborn curiosity and determination to create an LLM from scratch.

I hope this repository serves as a helpful reference and source of inspiration for researchers and open-source developers working on Korean NLP. If you find this project meaningful, please consider leaving a star (⭐) on GitHub. Thank you!
