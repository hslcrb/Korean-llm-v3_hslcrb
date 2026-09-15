import traceback
from pathlib import Path
from typing import List, Optional
import torch
from torch.utils.data import Dataset
from datasets import Dataset as HFDataset
from src.utils.logging_utils import logger


class LocalKoreanDataset(Dataset):
    def __init__(
        self,
        dataset_paths: List[str],
        tokenizer,
        max_len: int = 256,
        data_samples_per_dataset: Optional[int] = None
    ):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.samples = []

        logger.info("📚 Loading local datasets...")

        for dataset_path in dataset_paths:
            try:
                parquet_file = Path(dataset_path) / "data.parquet"
                if not parquet_file.exists():
                    logger.warning(f"Parquet file not found: {parquet_file}")
                    continue

                ds = HFDataset.from_parquet(str(parquet_file))

                if data_samples_per_dataset:
                    ds = ds.select(range(min(len(ds), data_samples_per_dataset)))

                texts = self._extract_texts(ds)
                self.samples.extend(texts)

                logger.info(f"✅ Loaded {len(texts)} samples from {Path(dataset_path).name}")

            except Exception as e:
                logger.error(f"❌ Error loading dataset from {dataset_path}: {e}")
                logger.error(f"   Full traceback:\n{traceback.format_exc()}")
                continue

        logger.info(f"✅ Total samples loaded: {len(self.samples)}")

    def _clean_text(self, value) -> str:
        if value is None:
            return ""
        if not isinstance(value, str):
            value = str(value)
        return value.strip()

    def _build_instruction_sample(self, instruction: str, input_text: str, output: str) -> str:
        instruction = self._clean_text(instruction)
        input_text = self._clean_text(input_text)
        output = self._clean_text(output)

        parts = []
        if instruction:
            parts.append(f"### 질문: {instruction}")
        if input_text:
            parts.append(f"### 입력: {input_text}")
        parts.append(f"### 응답: {output}")
        return "\n".join(parts)

    def _extract_texts(self, ds) -> List[str]:
        texts = []

        for item in ds:
            text = None

            if "text" in item and item["text"]:
                text = self._clean_text(item["text"])

            elif "instruction" in item and "output" in item:
                text = self._build_instruction_sample(
                    instruction=item.get("instruction", ""),
                    input_text=item.get("input", ""),
                    output=item.get("output", "")
                )

            elif "question" in item and "response" in item:
                question = self._clean_text(item.get("question", ""))
                response = self._clean_text(item.get("response", ""))
                system = self._clean_text(item.get("system_prompt", ""))

                if system:
                    text = f"### 시스템: {system}\n### 질문: {question}\n### 응답: {response}"
                else:
                    text = f"### 질문: {question}\n### 응답: {response}"

            elif "question" in item and "answer" in item:
                question = self._clean_text(item.get("question", ""))
                answer = self._clean_text(item.get("answer", ""))
                text = f"### 질문: {question}\n### 응답: {answer}"

            elif "prompt" in item and "response" in item:
                prompt = self._clean_text(item.get("prompt", ""))
                response = self._clean_text(item.get("response", ""))
                text = f"### 질문: {prompt}\n### 응답: {response}"

            if text and len(text) > 5:
                texts.append(text)

        return texts

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> torch.Tensor:
        text = self.samples[idx]

        try:
            eos_id = self.tokenizer.eos_token_id
            pad_id = self.tokenizer.pad_token_id

            if eos_id is None:
                raise ValueError("tokenizer.eos_token_id가 없습니다.")
            if pad_id is None:
                raise ValueError("tokenizer.pad_token_id가 없습니다.")

            encoded = self.tokenizer.encode(
                text,
                add_special_tokens=False,
                truncation=True,
                max_length=self.max_len - 1
            )
            encoded.append(eos_id)

            if len(encoded) < self.max_len:
                encoded += [pad_id] * (self.max_len - len(encoded))
            else:
                encoded = encoded[:self.max_len]
                encoded[-1] = eos_id

            return torch.tensor(encoded, dtype=torch.long)

        except Exception as e:
            logger.warning(f"Tokenization error: {e}")
            return torch.full(
                (self.max_len,),
                self.tokenizer.pad_token_id,
                dtype=torch.long
            )


def collate_fn(batch: List[torch.Tensor]) -> torch.Tensor:
    return torch.stack(batch)
