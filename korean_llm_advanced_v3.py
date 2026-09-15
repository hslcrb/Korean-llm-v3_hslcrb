"""
Korean LLM Advanced v3 메인 모듈 및 진입점 (main.py 역할)

기존 스크립트와의 100% 하위 호환성을 제공하며, 
CLI 명령행 인자 파싱 및 src/ 패키지 하위의 모듈화된 엔진을 구동합니다.
"""

import argparse
from src import __version__
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
    "__version__",
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
    "parse_args",
]


def parse_args() -> TrainingConfig:
    """CLI 명령행 인자 파싱 함수"""
    parser = argparse.ArgumentParser(
        description=f"Korean LLM Advanced {__version__} - Primary Training Entrypoint"
    )
    parser.add_argument("--batch-size", type=int, default=2, help="GPU당 배치 크기 (기본값: 2)")
    parser.add_argument("--max-steps", type=int, default=50000, help="최대 학습 스텝 수 (기본값: 50000)")
    parser.add_argument("--accumulation-steps", type=int, default=8, help="그래디언트 누적 스텝 (기본값: 8)")
    parser.add_argument("--learning-rate", type=float, default=5e-5, help="학습률 (기본값: 5e-5)")
    parser.add_argument("--warmup-steps", type=int, default=200, help="워밍업 스텝 수 (기본값: 200)")
    parser.add_argument("--eval-interval", type=int, default=10000, help="평가 및 체크포인트 주기 (기본값: 10000)")
    parser.add_argument("--resume", type=str, default="auto", help="'auto', 'latest', 또는 체크포인트 경로")
    parser.add_argument("--download-datasets", action="store_true", help="데이터셋 강제 재다운로드 여부")
    parser.add_argument("--samples-per-dataset", type=int, default=None, help="디버깅용 데이터셋당 샘플 제한 수")

    args, _ = parser.parse_known_args()

    resume_ckpt = None
    if args.resume == "auto":
        resume_ckpt = 'latest' if find_latest_checkpoint() else None
    elif args.resume.lower() == "latest":
        resume_ckpt = 'latest'
    elif args.resume:
        resume_ckpt = args.resume

    return TrainingConfig(
        batch_size=args.batch_size,
        accumulation_steps=args.accumulation_steps,
        max_steps=args.max_steps,
        warmup_steps=args.warmup_steps,
        learning_rate=args.learning_rate,
        eval_interval=args.eval_interval,
        resume_from_checkpoint=resume_ckpt,
        download_datasets=args.download_datasets,
        samples_per_dataset=args.samples_per_dataset,
    )


if __name__ == "__main__":
    config = parse_args()
    main(config)
