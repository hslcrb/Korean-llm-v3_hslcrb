# 🇰🇷 Korean LLM Advanced v3

<div align="center">

[![License: GPL-3.0](https://img.shields.io/badge/License-GPL%203.0-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-brightgreen)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C)](https://pytorch.org)
[![CUDA](https://img.shields.io/badge/CUDA-11.8%2B-76B900)](https://developer.nvidia.com/cuda-toolkit)
[![Model Size](https://img.shields.io/badge/Model-1.09B%20Parameters-orange)](#-핵심-사양)
[![VRAM](https://img.shields.io/badge/VRAM%20Usage-~9GB-red)](#-vram-점유율-비교)

**한국어 지시 이행 및 대화에 특화된 1.09B 파라미터 경량 LLM**  
*처음부터 끝까지 독자적으로 구현한 트랜스포머 구조 & VRAM 극대화 엔지니어링*

[English](README_EN.md) • [📖 개발 스토리](docs/DEVELOPMENT_STORY.md) • [🏛️ 아키텍처 가이드](docs/ARCHITECTURE.md) • [🚀 빠른 시작](#-빠른-시작) • [❓ FAQ](#-자주-묻는-질문-faq)

</div>

---

## 📖 프로젝트 개요

**Korean LLM Advanced v3**는 개인 개발 환경 및 중소규모 GPU에서도 효율적으로 사전학습 및 미세조정을 수행할 수 있도록 설계된 **한국어 특화 1.09B(10억 9천만) 파라미터 경량 언어모델**입니다.

중학교 2학년 개발자가 인공지능 기초부터 시작해 50M 프로토타입, 541M v1, 1.09B v2를 거쳐 완성한 풀스크래치 오픈소스 프로젝트로, **VRAM을 23GB에서 9GB로 60% 이상 절감**하여 단일 일반 소비자용 GPU에서도 학습이 가능하도록 최적화했습니다.

> 💡 **개발 비하인드**: 44,000 스텝의 학습과 버그 수정, 그리고 양자화 적용에 이르는 상세한 개발기는 [**개발 스토리(docs/DEVELOPMENT_STORY.md)**](docs/DEVELOPMENT_STORY.md)에서 확인하실 수 있습니다.

---

## 🌟 주요 특징

### 🎯 핵심 사양
| 항목 | 사양 | 비고 |
| :--- | :--- | :--- |
| **모델 파라미터** | **1.09B** (약 1,093M) | 임베딩 및 20개 트랜스포머 계층 |
| **은닉층 크기 (`dim`)** | 1,920 차원 | 헤드당 192차원 |
| **레이어 수 (`n_layers`)** | 20 개 | TransformerBlock 적층 |
| **어텐션 헤드 (`n_heads`)** | 10 개 | Multi-Head Attention |
| **FFN 은닉 차원 (`hidden_dim`)** | 4,800 차원 | SwiGLU Gated 피드포워드 계층 |
| **최대 컨텍스트 길이** | 2,048 토큰 | RoPE 주파수 버퍼 사전 계산 |
| **토크나이저 어휘집** | 128,256+ | `beomi/Llama-3-Open-Ko-8B` (`<|pad|>` 토큰 지원) |

---

### 🔧 5대 핵심 최적화 기법

1. **BF16 자동 혼합 정밀도 (Bfloat16 AMP)**
   - FP32 대비 가중치 및 활성화 메모리 **50% 절약** (12GB → 6GB).
   - FP16 대비 넓은 다이내믹 레인지를 보장하여 언더플로우/오버플로우 없이 안정적 학습.
2. **8비트 AdamW 옵티마이저 (`bitsandbytes`)**
   - 옵티마이저 상태 메모리 **75% 절감** (표준 2.2GB → 약 0.55GB).
3. **그래디언트 체크포인팅 (`torch.utils.checkpoint`)**
   - 순전파 활성화 텐서를 메모리에 유지하지 않고 역전파 시 재계산하여 활성화 VRAM 30~40% 감소.
4. **그래디언트 누적 (Gradient Accumulation)**
   - 마이크로 배치 2, 누적 스텝 8~32를 조합하여 VRAM 부담 없이 배치 크기 16~64의 안정적인 효과 실현.
5. **동적 양자화 지원 및 고속 어텐션 (SDPA + KV Cache)**
   - PyTorch 내장 `F.scaled_dot_product_attention` 커널 가속 및 추론 시 KV 캐시 재사용.

---

## 💾 VRAM 점유율 비교

<div align="center">

| 버전 | 파라미터 체급 | 평균 VRAM 소모량 | 적용된 최적화 기법 |
| :---: | :---: | :---: | :--- |
| **v1** | 541M | ~11 GB | 기본 FP32 학습 |
| **v2** | ~1.1B | ~23 GB | BF16 + 그래디언트 체크포인팅 |
| **v3 (현재)** ⭐ | **1.09B** | **~9 GB** ✨ | **BF16 + 8-bit AdamW + 체크포인팅 + 양자화 최적화** |

*v3은 v2와 동일한 1B 체급을 유지하면서 VRAM 점유율을 60% 이상 낮췄습니다.*

</div>

---

## 🚀 빠른 시작

### 1. 요구사항
- **Python**: 3.9 이상 (3.11 / 3.12 권장)
- **GPU**: NVIDIA GPU (VRAM 9GB 이상 권장, CPU 모드 추론 지원)
- **CUDA**: 11.8 이상

### 2. 설치

```bash
# 1. 저장소 클론
git clone https://github.com/seoan1024/korean-llm-v3.git
cd korean-llm-v3

# 2. 필수 의존성 패키지 일괄 설치
pip install -r requirements.txt
```

### 3. 학습 실행

```bash
# 기본 설정으로 학습 시작 (단일 메인 진입점)
python korean_llm_advanced_v3.py

# 하이퍼파라미터 커스텀 실행 (CLI 인자 지원)
python korean_llm_advanced_v3.py --batch-size 2 --accumulation-steps 8 --learning-rate 5e-5 --max-steps 50000
```

> 💡 **자동 데이터셋 다운로드**: 실행 시 `nlpai-lab/kullm-v2` 및 `beomi/KoAlpaca-v1.1a` 데이터셋을 자동으로 내려받아 로컬 Parquet으로 안전하게 캐싱합니다.

### 4. 실시간 GUI 모니터링
학습이 시작되면 자동으로 Tkinter & Matplotlib 기반의 모니터링 창이 구동됩니다:
- 📉 **실시간 Loss 커브 플롯**: 학습 손실값 추이를 실시간 그래프로 표시.
- 💬 **CPU 대화 테스트 창**: 최신 체크포인트를 비동기로 로드하여 학습 중간중간 실시간 텍스트 생성을 테스트할 수 있습니다.

---

## 🏗️ 프로젝트 구조

```
korean-llm-v3/
├── korean_llm_advanced_v3.py         # 🚀 [메인] CLI 인자 지원 및 하위 호환 단일 메인 진입점
├── requirements.txt                   # 필수 라이브러리 명세
├── .gitignore                         # 빌드/체크포인트/캐시 추적 제외
├── LICENSE                            # GNU GPL-3.0 라이선스
│
├── src/                               # 📦 모듈화된 핵심 소스코드
│   ├── __init__.py                    # __version__ = "v3.1.0" 및 주요 심볼 export
│   ├── config.py                      # 경로 상수 및 TrainingConfig 정의
│   ├── models/                        # 트랜스포머 아키텍처 (KoreanLLM, RMSNorm, RoPE 등)
│   ├── data/                          # DatasetManager, LocalKoreanDataset, collate_fn
│   ├── training/                      # train 루프 및 분산 학습 설정
│   ├── generation/                    # generate (KV Cache 텍스트 생성 엔진)
│   ├── gui/                           # TrainingMonitorGUI 모니터링 창
│   └── utils/                         # 로깅 및 체크포인트 I/O 유틸리티
│
├── docs/                              # 📚 상세 문서
│   ├── DEVELOPMENT_STORY.md           # [국문] 프로젝트 개발기 (중2 개발자의 도전)
│   ├── DEVELOPMENT_STORY_EN.md        # [영문] Project Development Journey
│   ├── ARCHITECTURE.md                # [국문] 모델 아키텍처 및 최적화 기술 가이드
│   └── ARCHITECTURE_EN.md             # [영문] Model Architecture & Optimization Guide
│
├── archive/                           # 🗄️ 레거시 스크립트 아카이브
│   ├── README.md                      # 아카이브 안내 (현재 실행 미사용, 원본 보존 목적)
│   └── korean_llm_advanced_v3tresure.py # 리팩토링 이전 원본 단일 스크립트
│
├── checkpoints/                       # 💾 저장된 학습 체크포인트 (.pth)
├── datasets/cache/                    # 📥 로컬 Parquet 데이터셋 캐시
└── logs/                              # 📝 학습 로그 및 loss_history.json
```

---

## 🔧 기술 스택

| 라이브러리 | 권장 버전 | 용도 |
| :--- | :--- | :--- |
| **PyTorch** | 2.0+ | 딥러닝 코어 프레임워크 및 SDPA 연산 |
| **Transformers** | 4.30+ | `Llama-3-Open-Ko-8B` 토크나이저 및 스케줄러 |
| **Datasets** | 2.10+ | 허깅페이스 한국어 데이터셋 스트리밍 및 로드 |
| **bitsandbytes** | 0.43+ | 8-bit AdamW 옵티마이저 및 양자화 |
| **pyarrow** | 12.0+ | 고속 로컬 Parquet 데이터셋 처리 |
| **matplotlib** | 3.7+ | 학습 손실값 실시간 시각화 플롯 |
| **tkinter** | 내장 | GUI 실시간 모니터링 및 대화 인터페이스 |
| **tqdm / pandas** | 최신 | 진행률 표시 및 데이터 처리 |

---

## 💡 코드 사용 예시

### 1. 체크포인트 로드 및 대화 생성
```python
import torch
from transformers import AutoTokenizer
from src.models import KoreanLLM
from src.generation import generate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("beomi/Llama-3-Open-Ko-8B")
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

# 모델 초기화
model = KoreanLLM(vocab_size=len(tokenizer), pad_token_id=tokenizer.pad_token_id).to(device)

# 체크포인트 로드
checkpoint = torch.load("checkpoints/korean_llm_50000.pth", map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# 텍스트 생성
response = generate(model, tokenizer, prompt="인공지능이란 무엇인가요?", max_tokens=128, device=device)
print("응답:", response)
```

### 2. 커스텀 학습 설정
```python
from src.config import TrainingConfig
from src.training import train

config = TrainingConfig(
    batch_size=2,
    accumulation_steps=16,
    max_steps=50000,
    learning_rate=3e-5,
    warmup_steps=300,
    use_bfloat16=True,
    resume_from_checkpoint="latest"
)

train(config)
```

---

## ❓ 자주 묻는 질문 (FAQ)

<details>
<summary><b>Q1. 모델 추론만 가볍게 테스트하고 싶습니다.</b></summary>
<p>
체크포인트 파일(.pth)이 있다면 상단의 '코드 사용 예시 1'과 같이 <code>KoreanLLM</code> 모델을 선언하고 가중치를 로드한 후 <code>generate()</code> 함수를 호출하면 간편하게 추론할 수 있습니다.
</p>
</details>

<details>
<summary><b>Q2. 보유 GPU의 VRAM이 8GB 이하입니다. 어떻게 조절해야 하나요?</b></summary>
<p>
다음 설정으로 VRAM 사용량을 추가로 낮출 수 있습니다:
1. <code>batch_size=1</code> 설정
2. <code>max_seq_len=256</code> 또는 <code>512</code>로 조정
3. <code>accumulation_steps=16</code> 이상으로 확대하여 유효 배치 크기 보정
</p>
</details>

<details>
<summary><b>Q3. 학습 도중 중단되었을 때 이어서 학습하려면 어떻게 하나요?</b></summary>
<p>
<code>python korean_llm_advanced_v3.py --resume latest</code>를 실행하면 <code>checkpoints/</code> 디렉토리에서 가장 최근 스텝 번호의 가중치를 자동 감지하여 옵티마이저 및 스케줄러 상태까지 완벽하게 복원합니다.
</p>
</details>

<details>
<summary><b>Q4. 다른 한국어 데이터셋을 추가하거나 변경하려면 어떻게 하나요?</b></summary>
<p>
<code>src/data/dataset_manager.py</code> 내의 <code>DATASETS_CONFIG</code> 리스트에 허깅페이스 데이터셋 이름과 split, text 키 목록을 추가하면 자동으로 다운로드 및 Parquet 변환이 적용됩니다.
</p>
</details>

<details>
<summary><b>Q5. Windows 환경에서 DataLoader 다중 워커 에러가 발생합니다.</b></summary>
<p>
Windows에서는 파이썬 멀티프로세싱 방식으로 인해 <code>num_workers</code>가 높을 경우 충돌이 일어날 수 있습니다. <code>TrainingConfig(num_workers=0)</code>으로 설정하여 단일 프로세스로 안정적으로 실행하십시오.
</p>
</details>

<details>
<summary><b>Q6. GPU가 없는 CPU 환경에서도 실행이 가능한가요?</b></summary>
<p>
네. 코드가 자동으로 <code>cuda</code> 사용 가능 여부를 감지하며, GPU가 없을 경우 CPU 모드로 전환됩니다. (다만 1.09B 모델 특성상 CPU 학습은 속도가 느릴 수 있으므로 테스트 및 추론 용도로 권장합니다.)
</p>
</details>

<details>
<summary><b>Q7. 생성되는 텍스트 품질을 높이려면 어떤 하이퍼파라미터를 만져야 하나요?</b></summary>
<p>
<code>generate()</code> 함수의 <code>temperature</code>(권장: 0.5~0.7), <code>top_p</code>(권장: 0.9~0.95), <code>repetition_penalty</code>(권장: 1.15~1.3) 값을 조정하여 문맥에 적절한 창의성과 정확도를 확보할 수 있습니다.
</p>
</details>

<details>
<summary><b>Q8. archive 디렉토리에 있는 파일은 무엇인가요?</b></summary>
<p>
<code>archive/</code> 폴더는 리팩토링 이전 초기 단일 파일 형태의 원본 스크립트를 영구 보존하기 위한 아카이브 저장소입니다. 현재 실행이나 학습에는 사용되지 않으며 개발 히스토리 보존을 위한 용도입니다.
</p>
</details>

<details>
<summary><b>Q9. 모델을 ONNX 형식으로 변환하여 내보낼 수 있나요?</b></summary>
<p>
네. PyTorch 표준 <code>torch.onnx.export</code> 함수를 사용하여 <code>KoreanLLM</code> 모델 인스턴스와 더미 입력 텐서를 통해 ONNX 그래프로 손쉽게 내보낼 수 있습니다.
</p>
</details>

<details>
<summary><b>Q10. 상용 프로젝트나 2차 연구에 활용할 수 있나요?</b></summary>
<p>
본 프로젝트는 <b>GNU General Public License v3.0 (GPL-3.0)</b>에 따라 배포됩니다. 오픈소스 라이선스 규정을 준수하는 범위 내에서 자유롭게 연구, 수정 및 활용이 가능합니다.
</p>
</details>

---

## 🛠️ 트러블슈팅

- **CUDA Out of Memory (OOM) 발생 시**:
  - `batch_size=1`로 축소하고 `accumulation_steps`를 2배로 증가시키세요.
  - `max_seq_len`을 필요에 맞게 축소하십시오.
- **bitsandbytes 로드 실패 시**:
  - Windows의 경우 `pip install bitsandbytes>=0.43.0` 공식 휠 버전을 사용하거나 최신 CUDA 드라이버 환경을 점검하십시오.
- **데이터셋 다운로드 네트워크 타임아웃 시**:
  - `DatasetManager`에 내장된 3단계 폴백 전략(일반 → 스트리밍 → 강제 재다운로드)이 순차 시도됩니다. 캐시 문제가 발생하면 `datasets/cache/` 디렉토리를 정리 후 재실행하십시오.

---

## 📜 라이선스 및 기여

- **License**: [GNU General Public License v3.0 (GPL-3.0)](./LICENSE)
- **개발자**: [seoan1024](https://github.com/seoan1024) (Contact: seoan102410@gmail.com)
- 버그 제보, 아키텍처 제안, 풀 리퀘스트(PR)는 언제나 환영합니다!

<div align="center">

**⭐ 이 프로젝트가 유익하셨다면 GitHub 상단의 Star를 눌러 응원해주세요! ⭐**

</div>
