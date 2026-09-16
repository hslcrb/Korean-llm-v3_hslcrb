"""
Korean LLM Advanced v3 메인 모듈 및 진입점 (main.py 역할)

기존 스크립트와의 100% 하위 호환성을 제공하며, 
CLI 명령행 인자 파싱 및 src/ 패키지 하위의 모듈화된 엔진을 구동합니다.
"""

import sys
import os

# Windows 콘솔 CP949 인코딩 충돌 방지 및 UTF-8 설정
if sys.platform == "win32":
    try:
        import ctypes
        ctypes.windll.kernel32.SetConsoleOutputCP(65001)
        ctypes.windll.kernel32.SetConsoleCP(65001)
    except Exception:
        pass

if sys.stdout is not None:
    try:
        if hasattr(sys.stdout, 'reconfigure'):
            sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        elif hasattr(sys.stdout, 'buffer'):
            import io
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
    except Exception:
        pass

if sys.stderr is not None:
    try:
        if hasattr(sys.stderr, 'reconfigure'):
            sys.stderr.reconfigure(encoding='utf-8', errors='replace')
        elif hasattr(sys.stderr, 'buffer'):
            import io
            sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace', line_buffering=True)
    except Exception:
        pass

os.environ["TRANSFORMERS_NO_TORCHVISION"] = "1"
os.environ["PYTHONIOENCODING"] = "utf-8"

import argparse
from typing import Optional, List, Tuple
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


def get_arg_parser() -> argparse.ArgumentParser:
    """CLI ArgumentParser 인스턴스를 반환합니다."""
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
    return parser


def parse_args(args_list: Optional[List[str]] = None) -> TrainingConfig:
    """CLI 명령행 인자 파싱 함수"""
    parser = get_arg_parser()
    args, _ = parser.parse_known_args(args_list)

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
    import traceback

    try:
        print("=" * 65)
        print(f"🇰🇷 Korean LLM Advanced {__version__}")
        print("   한국어 1.09B 경량 언어모델 - 메인 실행 엔진")
        print("=" * 65)

        # CLI 인자 없이 exe를 더블클릭한 대화형 환경인 경우 안내 메뉴 제공
        if len(sys.argv) == 1 and sys.stdin and sys.stdin.isatty():
            try:
                import torch
                if torch.cuda.is_available():
                    device_info = f"CUDA ({torch.cuda.get_device_name(0)})"
                else:
                    device_info = "CPU (GPU 미감지)"
            except Exception:
                device_info = "알 수 없음"

            print(f"💻 감지된 하드웨어 디바이스: {device_info}")
            print("-" * 65)
            print("[실행 모드를 선택하세요]")
            print(" 1. 모델 학습 시작 (기본 설정: 배치 2, 누적 8, 50,000 스텝)")
            print(" 2. 빠른 검증/테스트 학습 (배치 1, 100 스텝, 샘플 50개)")
            print(" 3. CLI 사용법 및 옵션 도움말 (--help)")
            print(" 0. 종료 (Exit)")
            print("-" * 65)

            try:
                choice = input("선택 번호 입력 [1/2/3/0] (기본값: 1): ").strip()
            except (EOFError, KeyboardInterrupt):
                choice = "0"

            if choice == "2":
                print("\n🚀 빠른 검증 모드로 학습을 시작합니다 (배치 1, 100 스텝, 샘플 50개)...")
                config = TrainingConfig(batch_size=1, max_steps=100, samples_per_dataset=50)
                main(config)
            elif choice == "3":
                print("\n[CLI 명령행 인자 도움말]")
                get_arg_parser().print_help()
            elif choice == "0":
                print("\n👋 프로그램을 종료합니다.")
                sys.exit(0)
            else:
                print("\n🚀 기본 설정으로 모델 학습을 시작합니다...")
                config = parse_args()
                main(config)
        else:
            config = parse_args()
            main(config)

    except KeyboardInterrupt:
        print("\n\n👋 사용자에 의해 실행이 중단되었습니다.")
    except SystemExit:
        pass
    except Exception as e:
        print("\n❌ 실행 중 오류가 발생했습니다:")
        traceback.print_exc()
        print("\n" + "=" * 65)
        print("💡 문제 해결 팁:")
        print("1. 인터넷 연결 및 Hugging Face 접근 상태를 확인하세요.")
        print("2. GPU VRAM이 부족할 경우 --batch-size 1 옵션을 사용하세요.")
        print("=" * 65)
    finally:
        if sys.stdin and sys.stdin.isatty():
            try:
                input("\n엔터 키를 누르면 창이 닫힙니다...")
            except (EOFError, KeyboardInterrupt):
                pass

