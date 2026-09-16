import json
import logging
import sys
from pathlib import Path
from typing import List, Dict, Any
from src.config import LOG_DIR, LOSS_HISTORY_FILE

# Windows 콘솔 CP949 충돌 방지 및 UTF-8 설정
if sys.platform == "win32":
    try:
        import ctypes
        ctypes.windll.kernel32.SetConsoleOutputCP(65001)
        ctypes.windll.kernel32.SetConsoleCP(65001)
    except Exception:
        pass

if sys.stdout is not None and hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
if sys.stderr is not None and hasattr(sys.stderr, 'reconfigure'):
    try:
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

# 로그 디렉토리 생성
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / "training.log", encoding='utf-8')
    ]
)
logger = logging.getLogger("KoreanLLM")

loss_history: List[Dict[str, Any]] = []

def save_loss_history(file_path: Path = LOSS_HISTORY_FILE):
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(loss_history, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.warning(f"Loss history save failed: {e}")

def load_loss_history(file_path: Path = LOSS_HISTORY_FILE) -> List[Dict[str, Any]]:
    global loss_history
    if file_path.exists():
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                loss_history = json.load(f)
            logger.info(f"✅ Loaded {len(loss_history)} previous loss records")
        except Exception as e:
            logger.warning(f"Failed to load loss history: {e}")
            loss_history = []
    return loss_history
