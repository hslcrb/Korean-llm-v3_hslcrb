from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# ==========================================
# 기본 경로 설정
# ==========================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent

LOG_DIR = PROJECT_ROOT / "logs"
LOSS_HISTORY_FILE = LOG_DIR / "loss_history.json"

DATASETS_DIR = PROJECT_ROOT / "datasets"
DATASETS_CACHE_DIR = DATASETS_DIR / "cache"
DATASETS_MANIFEST_FILE = DATASETS_DIR / "datasets_manifest.json"

CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"


# ==========================================
# 학습 하이퍼파라미터 및 설정
# ==========================================
@dataclass
class TrainingConfig:
    batch_size: int = 2
    max_steps: int = 50000
    accumulation_steps: int = 32
    learning_rate: float = 5e-5
    warmup_steps: int = 200
    checkpoint_interval: int = 100
    eval_interval: int = 500
    max_seq_len: int = 512
    num_workers: int = 4
    use_bfloat16: bool = True
    seed: int = 42
    resume_from_checkpoint: Optional[str] = None
    download_datasets: bool = False
    samples_per_dataset: Optional[int] = None
