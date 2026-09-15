"""
Korean LLM Advanced v3 메인 모듈 및 하위 호환성 래퍼

기존 스크립트와의 100% 하위 호환성을 제공하며, 
내부 구현은 src/ 패키지 하위의 모듈화된 코드를 호출합니다.
"""

from src.config import (
    LOG_DIR,
    LOSS_HISTORY_FILE,
    DATASETS_DIR,
    DATASETS_CACHE_DIR,
    DATASETS_MANIFEST_FILE,
    CHECKPOINTS_DIR,
    TrainingConfig,
)
from src.utils.logging_utils import (
    logger,
    loss_history,
    save_loss_history,
    load_loss_history,
)
from src.utils.checkpoint import (
    save_checkpoint,
    load_checkpoint,
    find_latest_checkpoint,
)
from src.models.modules import (
    RMSNorm,
    precompute_freqs_cis,
    apply_rotary_emb,
    SwiGLU,
    Attention,
    TransformerBlock,
)
from src.models.korean_llm import KoreanLLM
from src.data.dataset_manager import (
    DatasetManager,
    ensure_datasets_dir,
)
from src.data.dataset import (
    LocalKoreanDataset,
    collate_fn,
)
from src.generation.generator import generate
from src.gui.monitor import TrainingMonitorGUI
from src.training.trainer import (
    setup_distributed,
    train as main,
    gui_monitor,
)

__all__ = [
    "LOG_DIR",
    "LOSS_HISTORY_FILE",
    "DATASETS_DIR",
    "DATASETS_CACHE_DIR",
    "DATASETS_MANIFEST_FILE",
    "CHECKPOINTS_DIR",
    "TrainingConfig",
    "logger",
    "loss_history",
    "save_loss_history",
    "load_loss_history",
    "save_checkpoint",
    "load_checkpoint",
    "find_latest_checkpoint",
    "RMSNorm",
    "precompute_freqs_cis",
    "apply_rotary_emb",
    "SwiGLU",
    "Attention",
    "TransformerBlock",
    "KoreanLLM",
    "DatasetManager",
    "ensure_datasets_dir",
    "LocalKoreanDataset",
    "collate_fn",
    "generate",
    "TrainingMonitorGUI",
    "setup_distributed",
    "main",
    "gui_monitor",
]

if __name__ == "__main__":
    config = TrainingConfig(
        batch_size=2,
        accumulation_steps=8,
        max_steps=50000,
        warmup_steps=200,
        learning_rate=5e-5,
        eval_interval=10000,
        resume_from_checkpoint='latest' if find_latest_checkpoint() else None,
        download_datasets=False,
        samples_per_dataset=None,
    )
    main(config)
