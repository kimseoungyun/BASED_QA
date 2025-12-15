# 🧠 RLHF-RAG-BASED-QA  
**Reinforcement Learning from Human Feedback (RLHF) 기반 RAG QA 모델 학습 파이프라인**

본 프로젝트는 **Retrieval-Augmented Generation (RAG)** 환경에서  
대규모 언어 모델(LLM)을 **인간 피드백 기반 강화학습(RLHF)** 으로 정제하여  
질의응답(QA) 성능을 향상시키는 것을 목표로 합니다.

Supervised Fine-Tuning(SFT) → Reward Model(RM) → PPO → LoRA Merge의  
표준 RLHF 파이프라인을 구현하고,  
실험적으로 **freeze vs fine-tuning 전략의 차이**를 비교합니다.

---
## 🚀 Project Context

> **"From Zero to RLHF Pipeline in 7 Days"**

본 프로젝트는 **1주일(7 Days)**이라는 제한된 시간 내에 주제 선정부터 모델 학습, 파이프라인 구축까지 완료한 **Intensive Sprint 프로젝트**입니다.
특히 **RAG와 RLHF, Open Source LLM 튜닝을 처음 도입**하는 도전적인 상황에서도, SFT-RM-PPO로 이어지는 전체 학습 파이프라인을 성공적으로 구현하고 검증하는 데 집중했습니다.

* **수행 기간:** 2025.10.24 ~ 2025.10.31
* **핵심 성과:** RAG와 RLHF, Open Source LLM 튜닝을 처음 도입하여 SFT-RM-PPO 파이프라인 전체 구현
---

## 1. 프로젝트 목표 및 핵심 가치 (Goals & Core Values)

### 1.1 프로젝트 목표
- **고품질 QA 응답 생성**: RAG 기반 질의응답에서 일관성 있고 신뢰도 높은 응답 생성
- **인간 선호 반영**: 단순 정답률이 아닌 *인간이 선호하는 응답*을 학습
- **효율적 학습**: LoRA 및 부분 파인튜닝을 활용한 경량 RLHF 파이프라인 구현

### 1.2 핵심 아이디어
- **RLHF 적용**: 사람의 선택 데이터를 통해 보상 신호를 정의
- **PPO 기반 안정적 강화학습**
- **LoRA 병합(Merge)**을 통한 실사용 가능한 단일 모델 생성

---

## 2. 전체 파이프라인 개요 (Pipeline Overview)

[Dataset]<br>
   ↓<br>
[SFT (Supervised Fine-Tuning)]<br>
   ↓<br>
[RM (Reward Model)]<br>
   ↓<br>
[PPO (RL Fine-Tuning)]<br>
   ↓<br>
[LoRA Merge]<br>
   ↓<br>
[Final QA Model]<br>


### 2.1 SFT (Supervised Fine-Tuning)
- 인간 라벨이 포함된 QA 데이터셋으로 지도학습
- 기본 언어 모델이 질문-응답 구조를 학습

### 2.2 RM (Reward Model)
- 두 개의 응답 중 **더 인간적인 응답**을 선택하도록 학습
- PPO 단계에서 보상 함수 역할 수행

### 2.3 PPO (Reinforcement Learning)
- Reward Model의 점수를 보상으로 사용
- Proximal Policy Optimization(PPO)을 통해 모델 업데이트

### 2.4 MERGE (LoRA Merge)
- 학습된 LoRA adapter를 base model에 병합
- 단일 추론용 모델로 저장

---

## 3. 시스템 구조 및 폴더 구성 (Project Structure)

RLHF-RAG-BASED-QA/<br>
├── SFT.py        # Supervised Fine-Tuning<br>
├── RM.py         # Reward Model 학습<br>
├── PPO.py        # PPO 기반 RLHF 학습<br>
├── MERGE.py      # LoRA 병합 및 최종 모델 저장<br>
└── README.md<br>

---

## 4. Fine-Tuning 전략 비교 (핵심 실험 포인트)

### 🔹 기존 실험 (Baseline)
- 모든 pretrained 모델 가중치 **freeze**
- 학습 대상: **MLP / LoRA 헤드만 학습**
- 장점: 빠른 학습, 안정성
- 한계: 도메인 적응력 제한

### 🔹 Fine-Tuning 실험 (본 프로젝트 핵심)
- **상위 2개 Transformer 레이어 학습 허용**
- 고수준 의미 표현(semantic representation) 미세 조정
- 기대 효과:
  - 도메인 적응력 향상
  - QA 응답 품질 개선
  - 인간 선호 반영 강화

> 📌 본 README의 실험 설명은  
> **“상위 레이어 파인튜닝을 허용한 RLHF 실험”을 기준으로 작성되었습니다.**

---

## 5. 모델 및 학습 설정 (Model Configuration)

### 5.1 Base Model
- **Base LLM**: `meta-llama/Llama-3.2-1B`
- **Tokenizer**: HuggingFace Transformers

### 5.2 Optimization
- **RL Algorithm**: PPO
- **Fine-Tuning Method**: LoRA
- **Quantization**: BitsAndBytes (4bit, nf4)

---

## 6. 실험 의의 및 기대 효과

- RAG 환경에서 **RLHF 적용 가능성 검증**
- Freeze vs Partial Fine-Tuning 전략 비교
- 실제 서비스에 적용 가능한 **경량 RLHF 파이프라인 제시**

---

## 7. License
본 프로젝트는 **연구 및 교육 목적**으로 사용됩니다.  
사용된 모든 pretrained 모델은 Hugging Face 등 공개 라이선스를 따릅니다.

---

## 8. Retrospective & Challenges

### ⏱️ 1-Week Intensive Sprint
주제 선정부터 최종 모델 병합(Merge)까지 **단 1주일** 동안 진행된 프로젝트입니다. 짧은 기간 내에 전체 파이프라인을 완성하기 위해, 복잡한 모델 아키텍처 변경보다는 **안정적인 베이스라인 구축과 실험 파이프라인 자동화**에 주력했습니다.

### 💡 Tech Stack Expansion (First Step into RLHF & LLM)
기존의 정형 데이터 분석을 넘어, **Open Source LLM(Llama-3.2)**을 직접 핸들링하고 **RLHF(PPO)**와 **RAG**를 결합하는 시도를 **처음으로 수행**했습니다.
* **Challenge:** 강화학습(PPO)의 불안정한 학습 과정과 GPU 메모리 이슈.
* **Overcome:** `bitsandbytes` 양자화(Quantization)와 `LoRA`를 적극 활용하여 경량화된 학습 환경을 구축함으로써 하드웨어 제약을 극복했습니다.
