import os
import random
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
import bitsandbytes as bnb

from src.config import TrainingConfig, DATASETS_DIR, LOG_DIR
from src.utils.logging_utils import logger, loss_history, save_loss_history, load_loss_history
from src.utils.checkpoint import save_checkpoint, load_checkpoint, find_latest_checkpoint
from src.models.korean_llm import KoreanLLM
from src.data.dataset_manager import DatasetManager, ensure_datasets_dir
from src.data.dataset import LocalKoreanDataset, collate_fn
from src.generation.generator import generate
from src.gui.monitor import TrainingMonitorGUI

gui_monitor: Optional[TrainingMonitorGUI] = None


def setup_distributed(rank: int = 0, world_size: int = 1):
    random.seed(42 + rank)
    torch.manual_seed(42 + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42 + rank)


def train(config: TrainingConfig = TrainingConfig()):
    global gui_monitor

    setup_distributed()
    ensure_datasets_dir()
    load_loss_history()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    logger.info(f"📂 Datasets directory: {DATASETS_DIR.absolute()}")
    logger.info(f"📂 Logs directory: {LOG_DIR.absolute()}")

    # ============================================
    # 1. 토크나이저 로드
    # ============================================
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        "beomi/Llama-3-Open-Ko-8B",
        clean_up_tokenization_spaces=False
    )

    if tokenizer.pad_token is None or tokenizer.pad_token_id == tokenizer.eos_token_id:
        tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
        logger.info("✅ Added separate pad token: <|pad|>")

    logger.info(f"Tokenizer: vocab_size={len(tokenizer)}, eos_id={tokenizer.eos_token_id}, pad_id={tokenizer.pad_token_id}")

    # ============================================
    # 2. 데이터셋 다운로드 및 로드
    # ============================================
    logger.info("Setting up datasets...")
    manager = DatasetManager()

    dataset_paths = manager.get_or_download_all(force=config.download_datasets)

    if not dataset_paths:
        logger.error("❌ No datasets available!")
        return

    dataset = LocalKoreanDataset(
        dataset_paths=dataset_paths,
        tokenizer=tokenizer,
        max_len=config.max_seq_len,
        data_samples_per_dataset=config.samples_per_dataset
    )

    if len(dataset) == 0:
        logger.error("❌ Dataset is empty!")
        return

    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        shuffle=True,
        collate_fn=collate_fn,
        pin_memory=(device.type == "cuda")
    )

    # ============================================
    # 3. 모델 생성 (VRAM 최적화: BF16 변환)
    # ============================================
    logger.info("Creating model...")
    model_config = dict(
        vocab_size=len(tokenizer),
        pad_token_id=tokenizer.pad_token_id,
        dim=1920,
        n_layers=20,
        n_heads=10,
        max_seq_len=config.max_seq_len
    )
    # 모델을 처음부터 bfloat16으로 로드하여 VRAM 절반(약 6GB) 절약
    if device.type == 'cuda':
        model = KoreanLLM(**model_config).to(device).to(torch.bfloat16)
    else:
        model = KoreanLLM(**model_config).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model: {total_params / 1e6:.1f}M total params, {trainable_params / 1e6:.1f}M trainable")

    # ============================================
    # GUI 시작
    # ============================================
    def start_gui():
        global gui_monitor
        gui_monitor = TrainingMonitorGUI(tokenizer, device, model_config)
        gui_monitor.run()

    gui_thread = threading.Thread(target=start_gui, daemon=True)
    gui_thread.start()
    time.sleep(1.5)
    logger.info("🖥️  Monitoring GUI started (Loss graph + Chat)")

    # ============================================
    # 4. 옵티마이저와 스케줄러 (VRAM 최적화: 8-bit Optimizer)
    # ============================================
    try:
        optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=config.learning_rate)
    except Exception as e:
        logger.warning(f"bnb.optim.AdamW8bit initialization failed ({e}), falling back to torch.optim.AdamW")
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=config.max_steps
    )

    # BF16 사용 시 Scaler는 기본적으로 필요 없으나, 하위 호환성을 위해 유지
    scaler = torch.amp.GradScaler('cuda') if (device.type == 'cuda' and not config.use_bfloat16) else None

    # ============================================
    # 5. 체크포인트 로드
    # ============================================
    start_step = 0

    if config.resume_from_checkpoint:
        checkpoint_path = config.resume_from_checkpoint

        if checkpoint_path.lower() == 'latest':
            checkpoint_path = find_latest_checkpoint()
            if checkpoint_path is None:
                logger.warning("No checkpoint found, starting from scratch")

        if checkpoint_path and os.path.exists(checkpoint_path):
            logger.info(f"🔄 Loading checkpoint from: {checkpoint_path}")
            start_step = load_checkpoint(
                checkpoint_path,
                model,
                optimizer,
                scheduler,
                device
            )
            if gui_monitor:
                gui_monitor.notify_checkpoint(checkpoint_path)

    # ============================================
    # 6. 학습 루프
    # ============================================
    logger.info(f"🚀 Starting training from step {start_step}...")
    logger.info(f"📊 Dataset size: {len(dataset)} samples")
    logger.info(f"📊 Total batches per epoch: {len(loader)}")

    model.train()
    optimizer.zero_grad()

    running_loss = 0.0
    step = 0

    try:
        epoch = 0
        while True:
            epoch += 1
            logger.info(f"\n📍 Epoch {epoch}")

            for batch_idx, batch in enumerate(loader):
                actual_step = (step // config.accumulation_steps) + start_step

                if actual_step >= config.max_steps:
                    logger.info(f"Reached max steps ({config.max_steps}), stopping training")
                    break

                batch = batch.to(device)

                if device.type == 'cuda' and config.use_bfloat16:
                    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                        _, loss, _ = model(batch, labels=batch)
                        loss_scaled = loss / config.accumulation_steps

                    # bfloat16은 scaler.scale이 필요 없으므로 바로 backward
                    loss_scaled.backward()
                else:
                    _, loss, _ = model(batch, labels=batch)
                    loss_scaled = loss / config.accumulation_steps
                    loss_scaled.backward()

                running_loss += loss.item()

                if step % 4 == 0:
                    print(".", end="", flush=True)

                if (step + 1) % config.accumulation_steps == 0:
                    if device.type == 'cuda' and config.use_bfloat16:
                        # bfloat16은 unscale이 필요 없음
                        pass
                    elif scaler:
                        scaler.unscale_(optimizer)

                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                    if scaler and not config.use_bfloat16:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()

                    optimizer.zero_grad()
                    scheduler.step()

                    actual_step = (step + 1) // config.accumulation_steps + start_step
                    avg_loss = running_loss / config.accumulation_steps
                    lr = scheduler.get_last_lr()[0]

                    log_msg = f"[Step {actual_step:5d}] Loss: {avg_loss:.4f} | LR: {lr:.2e} | Tokens/step: {config.batch_size * config.max_seq_len}"
                    print(f"\n{log_msg}")
                    logger.info(log_msg)

                    loss_history.append({
                        "step": actual_step,
                        "loss": float(avg_loss),
                        "lr": float(lr),
                        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    })
                    save_loss_history()

                    if gui_monitor:
                        gui_monitor.notify_loss()

                    running_loss = 0.0

                    if actual_step > 0 and actual_step % 250 == 0:
                        logger.info(f"⏸️ {actual_step}스텝 도달: 5초간 휴식합니다...")
                        time.sleep(5)

                    if actual_step % config.eval_interval == 0:
                        logger.info("\n📝 Generating samples...")
                        prompts = [
                            "한국의 수도는",
                            "인공지능이란",
                            "안녕?"
                        ]
                        for prompt in prompts:
                            response = generate(
                                model, tokenizer, prompt=prompt,
                                max_tokens=50, temperature=0.7, top_p=0.95, device=device
                            )
                            logger.info(f"  Q: {prompt}\n  A: {response}")

                        checkpoint_path = f"checkpoints/korean_llm_{actual_step:05d}.pth"
                        save_checkpoint(model, optimizer, scheduler, actual_step, checkpoint_path)

                        if gui_monitor:
                            gui_monitor.notify_checkpoint(checkpoint_path)
                            gui_monitor.notify_log(f"체크포인트 저장됨: {Path(checkpoint_path).name}")

                step += 1

            if actual_step >= config.max_steps:
                break

    except KeyboardInterrupt:
        logger.info("\n⚠️ Training interrupted by user")
        actual_step = (step // config.accumulation_steps) + start_step
        checkpoint_path = f"checkpoints/korean_llm_interrupted_{actual_step:05d}.pth"
        save_checkpoint(model, optimizer, scheduler, actual_step, checkpoint_path)
        if gui_monitor:
            gui_monitor.notify_checkpoint(checkpoint_path)

    except Exception as e:
        logger.error(f"Training error: {e}", exc_info=True)

    logger.info("🎉 Training completed!")
    save_loss_history()

    if gui_monitor and gui_monitor.running:
        logger.info("GUI가 열려 있습니다. 창을 닫으면 종료됩니다.")
        while gui_monitor.running:
            time.sleep(1)
