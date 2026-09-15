# 🏛️ 모델 아키텍처 및 최적화 가이드 (Architecture Guide)

Korean LLM Advanced v3는 한국어 자연어 처리에 최적화된 **1.09B 디코더 온리(Decoder-Only) 트랜스포머 언어모델**입니다. 
본 문서는 모델의 수학적 구성 요소, 텐서 규격, 그리고 VRAM을 23GB에서 9GB로 절감한 최적화 엔지니어링 기법을 상세히 다룹니다.

---

## 1. 모델 세부 사양 (Model Specifications)

| 하이퍼파라미터 | 값 | 설명 |
| :--- | :--- | :--- |
| **파라미터 수 (Total Params)** | 약 1.09B (1,093M) | 임베딩 및 트랜스포머 전 계층 가중치 합계 |
| **은닉층 차원 (`dim`)** | 1,920 | 토큰 임베딩 및 어텐션 입력 벡터 차원 |
| **레이어 수 (`n_layers`)** | 20 | 적층된 트랜스포머 블록 수 |
| **어텐션 헤드 (`n_heads`)** | 10 | 멀티 헤드 어텐션 헤드 수 |
| **헤드당 차원 (`head_dim`)** | 192 (`1920 // 10`) | 개별 어텐션 헤드의 투영 차원 |
| **FFN 은닉 차원 (`hidden_dim`)** | 4,800 (`int(dim * 2.5)`) | SwiGLU 피드포워드 내부 확장 차원 |
| **최대 컨텍스트 길이 (`max_seq_len`)** | 2,048 토큰 | RoPE 주파수 버퍼 사전 계산 기준 길이 |
| **어휘집 크기 (`vocab_size`)** | 128,256+ | `beomi/Llama-3-Open-Ko-8B` 토크나이저 기준 (`<|pad|>` 특수 토큰 포함) |

---

## 2. 핵심 아키텍처 컴포넌트

### 1) RMSNorm (Root Mean Square Normalization)
기존 LayerNorm의 평균 감산(Mean Centering) 연산을 생략하고 분산의 제곱평근(RMS)만으로 정규화하여 연산 속도를 약 10~15% 개선했습니다:
$$\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d} \sum_{i=1}^{d} x_i^2 + \epsilon}} \odot \gamma$$

### 2) RoPE (Rotary Position Embedding)
절대 위치 임베딩 대신, Query와 Key 벡터에 회전 행렬을 적용하여 토큰 간의 상대적 거리를 내적에 자연스럽게 인코딩합니다. 이를 통해 긴 문맥에서도 위치 정보가 왜곡되지 않고 유지됩니다.

### 3) SwiGLU 피드포워드 네트워크
Gated Linear Unit 구조에 SiLU 활성화 함수를 결합하여 기존 ReLU나 GELU 대비 뛰어난 표현력을 제공합니다:
$$\text{SwiGLU}(x) = W_2 (\text{SiLU}(W_1 x) \odot W_3 x)$$

### 4) SDPA (Scaled Dot-Product Attention) 및 KV Cache
- **학습 시**: PyTorch의 내장 `F.scaled_dot_product_attention`을 활용하여 C++ 레벨에서 최적화된 메모리 접근 및 Causal Masking을 수행합니다.
- **추론 시**: 이전 스텝의 Key, Value 텐서를 재계산하지 않고 `kv_cache` 버퍼에 누적 결합하여 $O(N)$ 시간 복잡도로 고속 텍스트 생성을 수행합니다.

---

## 3. 메모리 최적화 엔지니어링 (VRAM 9GB 달성)

```
[VRAM 소모량 절감 흐름]
표준 FP32 학습 (~23GB)
  ↓ BF16 자동 혼합 정밀도 적용 (-50% 가중치/활성화 메모리)
  ↓ 8비트 AdamW 옵티마이저 (-75% 옵티마이저 상태 메모리)
  ↓ 그래디언트 체크포인팅 (-35% 순전파 텐서 메모리)
최종 VRAM: 약 9GB (단일 소비자용 GPU에서 학습 가능)
```

1. **BF16 혼합 정밀도 (Bfloat16 AMP)**:
   - FP16의 지수 비트(8-bit)와 동일한 동적 범위를 유지하여 언더플로우 위험 없이 FP32 대비 메모리를 절반으로 절약합니다.
2. **8-bit AdamW (`bitsandbytes`)**:
   - 옵티마이저의 1차, 2차 모멘텀 텐서를 비선형 블록 양자화(Block-wise Quantization)하여 메모리를 획기적으로 줄입니다.
3. **그래디언트 체크포인팅 (`torch.utils.checkpoint`)**:
   - 순전파 시 중간 레이어의 모든 활성화 값을 저장하지 않고 역전파 시 필요한 부분만 재계산(Recomputation)하여 VRAM 스파이크를 방지합니다.
4. **그래디언트 누적 (`Gradient Accumulation`)**:
   - `batch_size=2`와 `accumulation_steps=8`~`32` 설정을 통해 물리적 메모리 한계 내에서 16~64 크기의 유효 배치 효과를 실현합니다.
