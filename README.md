# Hermes-Driven Automated ML Evolution

A framework for **LLM-driven iterative model optimization** using a state machine architecture to solve the context compression problem in long Agentic workflows.

---

## The Core Engineering Problem

When using LLMs as autonomous agents for multi-round optimization tasks, a fundamental problem emerges: **context window compression**. As the conversation grows, the LLM progressively loses memory of earlier experiments — leading to duplicate work, contradictory decisions, and regression on previously explored strategies.

## Solution: State Machine + External Memory

This project implements a **state-machine-driven prompt architecture** with two key components:

1. **External Persistent Memory**: All experimental states, results, and decisions are written to local files (`evolution_report.md`, `train.py`) rather than relying on the LLM's internal context.
2. **Deterministic State Transitions**: A 4-state machine ensures each iteration reads prior state before making decisions, preventing context-blind regression.

```
[State 1: Baseline]
       ↓
[State 2: Feedback & Record]  ←─────────────────┐
       ↓                                         │
[State 3: Reflection & Evolution] ───────────────┘
       ↓ (on convergence or N = max)
[State 4: Final Synthesis]
```

### Why This Matters

Standard LLM agent loops fail at iteration 4–6 because the model "forgets" what it already tried. The state machine forces a **read-before-write** contract: the agent must read the external report before proposing any new experiment, making the loop robust to arbitrary context length.

---

## Repository Structure

```
hermes-ml-evolution/
├── README.md
├── skills/
│   └── kaggle.md                      ← Importable Hermes skill (plug-and-play)
├── v1_house_prices/                   ← Kaggle regression benchmark
│   ├── prompt_state_machine.md        ← V1: 10-round, submit every round
│   ├── prompt_v4_advanced.md          ← V4: EDA-first, convergence, CV Std tracking
│   └── evolution_report.md            ← Full 10-round experiment log with results
└── v2_fraud_detection/                ← Financial fraud detection (IEEE-CIS)
    ├── prompt_state_machine.md        ← Fraud-adapted prompt (AUC, time-based CV)
    └── PLAN.md                        ← Design doc: velocity features, imbalance handling
```

---

## V1: House Prices Benchmark Results

| Metric | Value |
|--------|-------|
| Dataset | Kaggle House Prices (1,460 train samples) |
| Rounds | 10 automated iterations |
| LLM Model | DS-V4-Flash |
| Baseline Score | 0.14640 RMSLE (Top ~90%) |
| **Final Score** | **0.12381 RMSLE (Top ~20%)** |
| Cost | ~¥1 RMB |
| Wall Time | 20 minutes end-to-end |

**Score trajectory:**

| Iter | Strategy | RMSLE |
|------|----------|-------|
| 1 | RandomForest baseline | 0.14640 |
| 3 | XGBoost + feature engineering | 0.12975 ⭐ |
| 6 | ElasticNet + log target | 0.12875 ⭐ |
| 7 | XGB + ElasticNet ensemble | 0.12551 ⭐ |
| 10 | 3-model ensemble (final) | **0.12381** 🏆 |

---

## V2: Fraud Detection (Extension)

Applies the same state machine framework to financial fraud detection (IEEE-CIS dataset), with domain-specific adaptations:
- Severe class imbalance handling (0.035% fraud rate)
- Transaction velocity feature engineering
- Time-based cross-validation (no data leakage)
- AUC-ROC as primary metric

See `v2_fraud_detection/` for the adapted prompt and design plan.

---

## Quick Start

1. Install [Hermes](https://hermes-agent.nousresearch.com) and configure your sandbox
2. Copy the prompt from `v1_house_prices/prompt_state_machine.md`
3. Paste into Hermes and run — the agent handles everything else

```bash
pip install kaggle pandas scikit-learn xgboost lightgbm
mkdir -p ~/.kaggle && cp kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
```
