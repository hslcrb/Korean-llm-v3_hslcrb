import os
from pathlib import Path
from typing import Optional, Union
import torch
import torch.nn as nn
from src.utils.logging_utils import logger
from src.config import CHECKPOINTS_DIR

def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    step: int,
    checkpoint_path: Union[str, Path]
):
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }

    torch.save(checkpoint, str(checkpoint_path))
    logger.info(f"✅ Checkpoint saved: {checkpoint_path}")

def load_checkpoint(
    checkpoint_path: Union[str, Path],
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device
) -> int:
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        return 0

    try:
        checkpoint = torch.load(str(checkpoint_path), map_location=device)

        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info("✅ Model state loaded")

        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logger.info("✅ Optimizer state loaded")

        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        logger.info("✅ Scheduler state loaded")

        start_step = checkpoint['step']
        logger.info(f"✅ Checkpoint loaded from step {start_step}")
        return start_step

    except Exception as e:
        logger.error(f"Error loading checkpoint: {e}")
        return 0

def find_latest_checkpoint(checkpoint_dir: Union[str, Path] = CHECKPOINTS_DIR) -> Optional[str]:
    checkpoint_path = Path(checkpoint_dir)
    if not checkpoint_path.exists():
        return None

    checkpoints = [f for f in os.listdir(checkpoint_path) if f.endswith('.pth')]
    if not checkpoints:
        return None

    try:
        checkpoints.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
    except (ValueError, IndexError):
        checkpoints.sort(key=lambda x: (checkpoint_path / x).stat().st_mtime)

    latest = checkpoints[-1]
    latest_path = str(checkpoint_path / latest)
    logger.info(f"Found latest checkpoint: {latest}")
    return latest_path
