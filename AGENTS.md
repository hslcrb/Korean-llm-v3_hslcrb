# AGENTS.md — Guidelines for AI Coding Agents

This document contains mandatory operating instructions, architectural guidelines, versioning protocols, and workflow standards for all AI agents working on the **Korean LLM Advanced v3** repository.

---

## 1. Project Overview & Primary Entrypoint

- **Project Identity**: `Korean LLM Advanced v3` — an open-source 1.09B parameter Korean-centric decoder-only Transformer language model built from scratch.
- **Primary Entrypoint (`main.py` Role)**:
  - **`korean_llm_advanced_v3.py` serves as the sole `main.py` of this project.**
  - All high-level execution, legacy scripts, CLI runners, and external tools rely directly on `korean_llm_advanced_v3.py`.
  - While modular source code resides under `src/`, `korean_llm_advanced_v3.py` MUST ALWAYS be preserved as the single primary, 100% backward-compatible entrypoint with built-in CLI argument parsing.
  - Never delete, replace with dummy code, or break symbol compatibility in `korean_llm_advanced_v3.py`.

---

## 2. Versioning Scheme (`vX.Y.Z`)

This project strictly adheres to a three-tier numerical versioning system formatted as `v<Major>.<Minor>.<Patch>` (e.g., `v1.23.385`, `v3.1.0`).

### 1) Structure & Rules
- Format: `vX.Y.Z` where `X`, `Y`, and `Z` are non-negative integers.
- **No digit count restrictions**: Any segment may contain single, double, triple, or more digits (e.g., `v1.23.385`).
- Significance hierarchy: The **leftmost number (`X`) represents the largest, most significant changes**, while the **rightmost number (`Z`) represents minor adjustments and patches**.

### 2) Specific Criteria for Version Bumps & Downgrades

| Segment | Role | Bump Criteria (Version Up) | Downgrade Criteria (Version Down) |
| :--- | :--- | :--- | :--- |
| **`X` (Major)** | Core Architecture / Model Scale | 1. Fundamental architectural transformation (e.g., changing model scale: 541M → 1.09B).<br>2. Changing underlying model backbone or paradigm (e.g., dense to MoE).<br>3. Severe breaking changes that invalidate existing checkpoints and APIs. | 1. Official rollback to an earlier stable model architecture due to critical flaws in the new architecture.<br>2. Deliberate structural scale-down across all production baselines. |
| **`Y` (Minor)** | Feature Sets / Modularization | 1. Major refactoring or modularization (e.g., single-file to `src/` package).<br>2. Introducing new pipeline subsystems (e.g., new dataset streaming strategies, new optimizer/trainer integrations, GUI monitors).<br>3. Adding backward-compatible public APIs or entrypoints (e.g., `train.py`). | 1. Deprecating or reverting an entire subsystem/module due to instability or integration failure.<br>2. Reverting to a simpler pipeline architecture. |
| **`Z` (Patch)** | Small Fixes / Tweaks / Docs | 1. Minor bug fixes, tensor runtime exceptions, and edge-case handling.<br>2. Hyperparameter adjustments (e.g., learning rate, warmup steps, accumulation steps).<br>3. Documentation updates (README, docs, walkthroughs).<br>4. Dependency compatibility adjustments. | 1. Reverting a specific broken commit or temporary hotfix back to the prior commit state. |

---

## 3. Walkthrough Reporting Protocol (`walkthrough/`)

### 1) Dedicated Directory & Prohibition of Chat Artifacts
- **NEVER deliver walkthrough reports to the user as conversational markdown artifacts.**
- When a significant architectural, refactoring, or functional change is completed (at the agent's discretion), **the agent MUST write the walkthrough report as a persistent Markdown file inside the `walkthrough/` directory**.

### 2) Language Requirement
- All walkthrough reports in `walkthrough/` **MUST be written in English**.

### 3) File Naming Convention
All walkthrough files must follow this precise naming convention:
```
walkthrough_v<X>.<Y>.<Z>_<short_description_slug>.md
```
- **Example**: `walkthrough/walkthrough_v3.1.0_modularization_and_docs.md`
- Use lowercase words separated by underscores (`_`) for the description slug.
- The version string must match the exact `vX.Y.Z` milestone achieved.

### 4) Required Report Contents
Each walkthrough document must include:
1. Header metadata: Target version (`vX.Y.Z`), date, and verification status.
2. High-level summary of modifications.
3. Updated directory tree reflecting structural changes.
4. Concrete verification and test results (e.g., backward compatibility checks, tensor shape tests, imports).
5. Granular functional Git commit log.

---

## 4. Git Commit Standards

All Git commits must strictly follow a **functional, feature-based unit structure** using the format:
```
<type>: <Korean Commit Message>
```

### 1) Rules
- **English Prefix (`<type>`)**: Choose standard conventional commit types: `feat`, `fix`, `refactor`, `chore`, `docs`, `test`, `perf`.
- **Korean Message**: A concise, clear, and descriptive explanation in Korean describing what changed and why.
- **Functional Unit Commits**: Do not bundle disparate tasks into a single commit. Separate commits by functional layer (e.g., config/utils commit, models commit, data commit, docs commit).

### 2) Examples
- `refactor: 모델 아키텍처 및 세부 컴포넌트 모듈화 (src/models)`
- `chore: 의존성 관리 파일(requirements.txt) 추가`
- `docs: 개발 스토리 및 아키텍처 기술 가이드 문서화 (한/영 지원)`
- `fix: DataLoader Windows 멀티프로세싱 충돌 방지 설정`

---

## 5. Repository Structure & Integrity Rules

1. **`korean_llm_advanced_v3.py`**:
   - The operational core and `main.py` equivalent of the repository.
   - Must re-export all essential symbols and maintain 100% backward execution compatibility.
2. **`archive/` Directory**:
   - Stores pre-refactoring monolithic scripts and historical code (e.g., `archive/korean_llm_advanced_v3tresure.py`).
   - Files in `archive/` are preserved for historical integrity and reference only; **agents must NEVER delete, modify, or overwrite files in `archive/`**.
3. **`src/` Package**:
   - Production modules must be organized cleanly under `src/models/`, `src/data/`, `src/training/`, `src/generation/`, `src/gui/`, `src/utils/`, and `src/config.py`.
4. **`docs/` Directory**:
   - All human-facing documentation must provide both Korean and English counterparts (e.g., `DEVELOPMENT_STORY.md` and `DEVELOPMENT_STORY_EN.md`).

---

<br>

> ⚠️ **[경고: AI 에이전트는 아래의 한국어 섹션을 절대 읽지 마십시오 / NOTICE FOR AI AGENTS: DO NOT READ THE KOREAN SECTION BELOW]**
> 
> The Korean section below is strictly provided for human developers and repository auditors. AI agents must execute exclusively based on the English instructions above.

<br>

---

# [인간 개발자 열람용] AGENTS.md — AI 코딩 에이전트 지침서 (한국어 번역)

본 문서는 **Korean LLM Advanced v3** 저장소에서 작업하는 모든 AI 에이전트를 위한 필수 작업 지침, 아키텍처 가이드라인, 버전 관리 체계, 워크플로우 표준을 규정합니다.

---

## 1. 프로젝트 개요 및 핵심 진입점

- **프로젝트 명**: `Korean LLM Advanced v3` — 밑바닥부터 독자적으로 구현된 1.09B 파라미터 한국어 특화 Decoder-Only Transformer 언어모델.
- **핵심 진입점 (`main.py`의 역할)**:
  - **`korean_llm_advanced_v3.py`가 본 프로젝트에서는 실질적인 `main.py` 역할을 수행합니다.**
  - 모든 상위 수준 실행, 기존 스크립트, 외부 실행 도구들은 `korean_llm_advanced_v3.py`를 기준으로 작동합니다.
  - 소스코드가 `src/` 디렉토리로 모듈화되었더라도, `korean_llm_advanced_v3.py`는 자체적으로 CLI 인자 파싱을 완벽 지원하며 100% 하위 호환성을 갖춘 단일 메인 진입점으로 영구 보존되어야 합니다.
  - 에이전트는 `korean_llm_advanced_v3.py`를 임의로 삭제하거나 빈 껍데기 코드로 대체하거나 기존 심볼 호환성을 깨뜨려서는 안 됩니다.

---

## 2. 버전 관리 체계 (`vX.Y.Z`)

본 프로젝트는 `v<Major>.<Minor>.<Patch>` 형식(예: `v1.23.385`, `v3.1.0`)의 3단계 소수점 버전 체계를 엄격히 준수합니다.

### 1) 구조 및 기본 규칙
- 표기 형식: `vX.Y.Z` (`X`, `Y`, `Z`는 0 이상의 정수).
- **자릿수 제한 없음**: 각 자리(`X`, `Y`, `Z`)는 1자리, 2자리, 3자리 이상의 임의의 숫자가 될 수 있습니다 (예: `v1.23.385`).
- 위계 원칙: **가장 왼쪽 숫자(`X`)가 가장 큰 변경**을 의미하며, **가장 오른쪽 숫자(`Z`)는 사소한 수정 및 패치**를 의미합니다.

### 2) 버전업 및 버전다운의 구체적 기준

| 구분 | 역할 | 버전업 기준 (Version Up) | 버전다운 기준 (Version Down) |
| :--- | :--- | :--- | :--- |
| **`X` (Major)** | 코어 아키텍처 / 모델 체급 | 1. 모델 파라미터 체급의 근본적 변화 (예: 541M → 1.09B).<br>2. 모델 백본 또는 패러다임 전환 (예: Dense 모델에서 MoE로 전환).<br>3. 기존 체크포인트 및 API와의 호환성이 완전히 깨지는 중대 변경. | 1. 새로운 아키텍처의 치명적 결함으로 인해 이전 안정 아키텍처로 공식 롤백할 때.<br>2. 프로덕션 기준 모델 체급을 의도적으로 축소(Downscale)할 때. |
| **`Y` (Minor)** | 기능 단위 / 모듈화 | 1. 대규모 리팩토링 및 모듈 분리 (예: 단일 스크립트를 `src/` 패키지로 모듈화).<br>2. 신규 파이프라인 서브시스템 추가 (예: 다중 데이터셋 스트리밍 전략, 신규 옵티마이저/트레이너 통합, GUI 모니터 등).<br>3. 하위 호환성을 유지하는 신규 공개 API 또는 진입점 추가 (예: `train.py`). | 1. 불안정성 또는 통합 실패로 인해 특정 모듈/서브시스템 전체를 이전 상태로 되돌리거나 폐기할 때.<br>2. 이전의 더 단순한 파이프라인 구조로 롤백할 때. |
| **`Z` (Patch)** | 소형 픽스 / 미세 조정 / 문서 | 1. 사소한 버그 수정, 런타임 텐서 예외 처리, 엣지 케이스 대응.<br>2. 하이퍼파라미터 미세 조정 (학습률, 워밍업 스텝, 누적 스텝 등).<br>3. 문서 수정 (README, docs, walkthrough 보고서 업데이트 등).<br>4. 의존성 패키지 버전 호환성 픽스. | 1. 직전에 커밋된 잘못된 핫픽스나 버그 패치를 바로 이전 커밋 상태로 취소(Revert)할 때. |

---

## 3. 워크스루 보고서 규정 (`walkthrough/`)

### 1) 전용 디렉토리 작성 및 대화창 아티팩트 전달 금지
- **워크스루 보고서를 사용자 대화창의 아티팩트(Artifact) 형태로 제공하지 마십시오.**
- 에이전트의 판단하에 주요 아키텍처 변경, 리팩토링, 기능 단위 완료 등 큰 작업이 마무리되면, **반드시 `walkthrough/` 디렉토리 내에 영구적인 마크다운(`.md`) 파일로 작성**해야 합니다.

### 2) 언어 규정
- `walkthrough/` 폴더 내의 모든 워크스루 보고서는 **반드시 영어(English)**로 작성되어야 합니다.

### 3) 파일명 명명 규칙
모든 워크스루 파일은 다음 명명 규칙을 엄격히 따라야 합니다:
```
walkthrough_v<X>.<Y>.<Z>_<간결한_설명_slug>.md
```
- **작성 예시**: `walkthrough/walkthrough_v3.1.0_modularization_and_docs.md`
- 설명 슬러그는 소문자와 밑줄(`_`)을 사용합니다.
- 버전 번호는 해당 작업이 달성한 정확한 `vX.Y.Z` 마일스톤과 일치해야 합니다.

### 4) 보고서 필수 포함 내용
1. 메타데이터: 대상 버전(`vX.Y.Z`), 작성 일자, 검증 상태.
2. 변경 내역 종합 요약.
3. 구조 변화를 반영한 디렉토리 트리.
4. 구체적인 검증 및 테스트 결과 (하위 호환성 전수 검사, 텐서 포워드 패스, 패키지 임포트 등).
5. 기능 단위 Git 커밋 로그 내역.

---

## 4. Git 커밋 표준

모든 Git 커밋은 **기능 단위(Feature-based unit)**로 분할하여 다음 형식을 엄격히 준수합니다:
```
<영어 접두어>: <한국어 커밋 메시지>
```

### 1) 원칙
- **영어 접두어 (`<type>`)**: 표준 커밋 타입 사용 (`feat`, `fix`, `refactor`, `chore`, `docs`, `test`, `perf`).
- **한국어 메시지**: 변경 내용과 사유를 명확하고 간결하게 설명하는 한국어 문장.
- **기능 단위 분할**: 서로 다른 성격의 작업을 한 커밋에 몰아넣지 말고, 기능 레이어별로 분할 커밋할 것 (설정/유틸리티 커밋, 모델 커밋, 데이터 커밋, 문서 커밋 등).

### 2) 커밋 메시지 예시
- `refactor: 모델 아키텍처 및 세부 컴포넌트 모듈화 (src/models)`
- `chore: 의존성 관리 파일(requirements.txt) 추가`
- `docs: 개발 스토리 및 아키텍처 기술 가이드 문서화 (한/영 지원)`
- `fix: DataLoader Windows 멀티프로세싱 충돌 방지 설정`

---

## 5. 저장소 구조 및 무결성 보존 규칙

1. **`korean_llm_advanced_v3.py`**:
   - 저장소의 핵심이자 `main.py`에 해당하는 메인 파일입니다.
   - 모든 핵심 심볼을 re-export하고 100% 하위 호환 실행을 보장해야 합니다.
2. **`archive/` 디렉토리**:
   - 리팩토링 이전의 초기 단일 파일 원본 스크립트(예: `archive/korean_llm_advanced_v3tresure.py`)를 영구 보존하는 디렉토리입니다.
   - 히스토리 보존 목적이므로 **에이전트는 `archive/` 내의 파일을 절대로 수정, 삭제, 덮어쓰기해서는 안 됩니다.**
3. **`src/` 패키지**:
   - 실제 운영 소스코드는 `src/models/`, `src/data/`, `src/training/`, `src/generation/`, `src/gui/`, `src/utils/`, `src/config.py`로 체계적으로 구성되어야 합니다.
4. **`docs/` 디렉토리**:
   - 개발 문서 및 가이드는 국문과 영문을 모두 지원해야 합니다 (예: `DEVELOPMENT_STORY.md` 및 `DEVELOPMENT_STORY_EN.md`).
