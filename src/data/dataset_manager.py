import json
import hashlib
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional
from datasets import load_dataset, Dataset as HFDataset
from src.config import DATASETS_DIR, DATASETS_CACHE_DIR, DATASETS_MANIFEST_FILE
from src.utils.logging_utils import logger


def ensure_datasets_dir():
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    DATASETS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"✅ Datasets directory ready: {DATASETS_DIR.absolute()}")


class DatasetManager:
    DATASETS_CONFIG = [
        {
            "name": "nlpai-lab/kullm-v2",
            "config": None,
            "split": "train",
            "text_keys": ["instruction", "input", "output"]
        },
        {
            "name": "beomi/KoAlpaca-v1.1a",
            "config": None,
            "split": "train",
            "text_keys": ["instruction", "input", "output"]
        }
    ]

    def __init__(self, cache_dir: Path = DATASETS_CACHE_DIR):
        self.cache_dir = cache_dir
        ensure_datasets_dir()
        self.manifest = self._load_manifest()

    def _load_manifest(self) -> Dict:
        if DATASETS_MANIFEST_FILE.exists():
            with open(DATASETS_MANIFEST_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}

    def _save_manifest(self):
        with open(DATASETS_MANIFEST_FILE, 'w', encoding='utf-8') as f:
            json.dump(self.manifest, f, indent=2, ensure_ascii=False)

    def _get_dataset_hash(self, config: Dict) -> str:
        config_str = json.dumps(config, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:8]

    def download_dataset(self, config: Dict, force: bool = False) -> Optional[str]:
        dataset_hash = self._get_dataset_hash(config)
        dataset_name = config["name"]

        if dataset_hash in self.manifest and not force:
            cached_path = self.manifest[dataset_hash].get("path")
            if cached_path and Path(cached_path).exists():
                logger.info(f"✅ Using cached dataset: {dataset_name}")
                return cached_path

        logger.info(f"📥 Downloading {dataset_name}...")

        last_error = None
        error_details = []

        strategies = [
            {"name": "standard", "streaming": False, "force_redownload": False},
            {"name": "streaming", "streaming": True, "force_redownload": False},
            {"name": "force_redownload", "streaming": False, "force_redownload": True},
        ]

        for attempt, strategy in enumerate(strategies, 1):
            try:
                logger.info(f"🔄 Attempt {attempt}/{len(strategies)} - Strategy: {strategy['name']}")

                load_kwargs = {
                    "path": config["name"],
                    "split": config["split"],
                    "cache_dir": str(self.cache_dir),
                }

                if config.get("config"):
                    load_kwargs["name"] = config["config"]

                if strategy["streaming"]:
                    load_kwargs["streaming"] = True

                if strategy["force_redownload"]:
                    load_kwargs["download_mode"] = "force_redownload"

                ds = load_dataset(**load_kwargs)

                if strategy["streaming"]:
                    logger.info("📥 Converting streaming dataset to regular dataset...")
                    ds = HFDataset.from_list(list(ds))

                local_path = self.cache_dir / f"{dataset_hash}"
                local_path.mkdir(exist_ok=True, parents=True)

                ds.to_parquet(str(local_path / "data.parquet"))

                self.manifest[dataset_hash] = {
                    "name": dataset_name,
                    "config": config,
                    "path": str(local_path),
                    "num_examples": len(ds),
                    "download_strategy": strategy["name"]
                }
                self._save_manifest()

                logger.info(f"✅ Dataset saved: {local_path} ({len(ds)} examples) via {strategy['name']}")
                return str(local_path)

            except Exception as e:
                last_error = e
                error_type = type(e).__name__
                error_msg = str(e)
                tb_str = traceback.format_exc()

                error_details.append({
                    "attempt": attempt,
                    "strategy": strategy["name"],
                    "error_type": error_type,
                    "error_msg": error_msg,
                    "traceback": tb_str
                })

                logger.warning(f"⚠️ Attempt {attempt} failed ({strategy['name']})")
                logger.warning(f"   ↳ Error Type : {error_type}")
                logger.warning(f"   ↳ Error Msg  : {error_msg}")

                if attempt < len(strategies):
                    wait_time = attempt * 2
                    logger.info(f"⏳ Waiting {wait_time} seconds before next attempt...")
                    time.sleep(wait_time)

        logger.error("=" * 80)
        logger.error(f"❌ Failed to download {dataset_name} after {len(strategies)} attempts")
        logger.error("=" * 80)

        for detail in error_details:
            logger.error(f"[Attempt {detail['attempt']}] Strategy: {detail['strategy']}")
            logger.error(f"  - Type   : {detail['error_type']}")
            logger.error(f"  - Message: {detail['error_msg']}")
            logger.error(f"  - Traceback:\n{detail['traceback']}")
            logger.error("-" * 60)

        logger.error(f"📌 Last error summary: {type(last_error).__name__}: {last_error}")
        logger.error("=" * 80)
        return None

    def get_or_download_all(self, force: bool = False) -> List[str]:
        paths = []
        failed = []

        for config in self.DATASETS_CONFIG:
            path = self.download_dataset(config, force=force)
            if path:
                paths.append(path)
            else:
                failed.append(config["name"])

        logger.info(f"✅ Ready with {len(paths)} datasets")

        if failed:
            logger.warning("=" * 60)
            logger.warning(f"⚠️ 다음 데이터셋 다운로드 실패 ({len(failed)}개):")
            for name in failed:
                logger.warning(f"   - {name}")
            logger.warning("=" * 60)

        return paths
