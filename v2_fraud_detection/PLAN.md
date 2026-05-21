# V2 Design Plan — Financial Fraud Detection

## Why This Extension Matters

House Prices proves the state machine architecture works. This extension applies the same framework to a **domain-relevant, production-grade problem**: financial fraud detection on the IEEE-CIS dataset.

Key differences that make this harder and more interesting:

| Dimension | House Prices (V1) | Fraud Detection (V2) |
|-----------|-------------------|----------------------|
| Dataset size | 1,460 rows | 590,540 transactions |
| Class balance | N/A (regression) | 0.035% fraud rate (extreme imbalance) |
| Eval metric | RMSLE | AUC-ROC + F1 @ threshold |
| Feature engineering | Area/quality cross | Velocity, device graph, time patterns |
| Validation strategy | Random 5-fold | Time-based split (no leakage) |
| Core challenge | Log-linear relationship | Rare event detection under distribution shift |

---

## Dataset

**IEEE-CIS Fraud Detection** (Kaggle)
- `train_transaction.csv`: 590k rows × 394 features (TransactionDT, TransactionAmt, card/addr/email features, V1-V339)
- `train_identity.csv`: 144k rows × 41 features (device type, browser, OS, id_ features)
- Fraud rate: ~3.5% in training, naturally imbalanced
- Download: `kaggle competitions download -c ieee-fraud-detection`

---

## Adapted State Machine Design

### State 0 — EDA (added vs V1)

V2 adds a dedicated EDA state before baseline, because fraud data has non-obvious structure:

```
EDA checklist:
- Class imbalance ratio
- TransactionDT range → derive hour_of_day, day_of_week, days_since_start
- TransactionAmt distribution (log transform?)
- Identity join coverage rate (how many transactions have identity data?)
- Missing value patterns: V-features (Vesta engineered), card/addr features
- Top categorical features by fraud rate: ProductCD, card4, card6, P_emaildomain
- Velocity patterns: same card in short time window
```

### State 1 — Baseline

```python
# Minimal baseline: LightGBM with class_weight, no feature engineering
# Key: use time-based split, NOT random split
# train: first 80% by TransactionDT, val: last 20%

import lightgbm as lgb
from sklearn.metrics import roc_auc_score

model = lgb.LGBMClassifier(
    n_estimators=500,
    learning_rate=0.05,
    class_weight='balanced',  # handles imbalance
    random_state=42
)
```

**Why LightGBM not RandomForest for baseline?**
- 590k rows × 400 features: RandomForest is too slow
- LightGBM handles missing values natively (V-features have heavy NAs)
- class_weight='balanced' is the simplest imbalance handling

### States 2-3 — Evolution Hypotheses

Ordered by expected impact:

#### High Impact
1. **Transaction velocity features**: count/sum/mean of transactions per card in past 1h/1d/7d window
   - `card1_count_1d`, `card_amt_mean_7d`, etc.
   - Why: fraud patterns cluster in time — a stolen card is used repeatedly in short windows
   
2. **Time-based features**: hour_of_day, day_of_week, is_weekend, days_since_start
   - Why: fraud has strong temporal patterns (late night, specific day-of-week peaks)

3. **Identity features join**: Left join train_identity on TransactionID
   - DeviceType, DeviceInfo, browser UA encoding
   - Why: device fingerprinting is core to fraud detection

#### Medium Impact
4. **Target encoding for high-cardinality categoricals**: card1, addr1, P_emaildomain
   - Why: OHE explodes dimensionality; mean-target encoding captures fraud rate per entity
   - Risk: data leakage → use out-of-fold encoding only

5. **Aggregation features by (card1, card2) pair**: unique merchants, countries, amounts
   - Why: card-level behavior fingerprint

6. **SMOTE / undersampling experiments**
   - Why: 0.035% fraud rate may under-represent minority class in gradient updates
   - Risk: SMOTE on tabular data often hurts more than helps in practice; try scale_pos_weight first

#### Lower Impact
7. **XGBoost vs LightGBM comparison**
   - LightGBM usually wins on large sparse data, but worth validating

8. **Ensemble: LightGBM + XGBoost weighted average**

9. **Threshold optimization**: AUC is the metric, but business deploys at a threshold
   - Find optimal F1 threshold, log alongside AUC

### State 4 — Final Synthesis

Output format (extends V1):
```markdown
## Final Summary
- Best AUC: X.XXXXX
- Best F1 @ threshold T: X.XXX
- Most impactful feature groups: [velocity | time | identity | ...]
- Top 10 features by importance
- Lessons learned for production fraud models
```

---

## Key Prompt Adaptations from V1

```diff
- 评价准则：Public RMSLE（越低越好）
+ 评价准则：AUC-ROC（越高越好），同时记录 F1-score @ optimal threshold

- 使用 Kaggle CLI 每轮提交
+ 不每轮提交（节省成本），改为本地 time-based CV 验证
+ 仅最终最优模型提交

- 5-Fold 随机 CV
+ 时序 CV: train on TransactionDT < T, validate on T ≤ TransactionDT < T+window
+ 原因: 随机 CV 会导致时间泄漏，高估真实泛化性能

- 基线: RandomForestRegressor
+ 基线: LightGBMClassifier(class_weight='balanced')
+ 原因: 数据量大 + 类别不平衡 + V特征大量缺失值
```

---

## Connection to Production Fraud Systems

This experiment is a controlled simulation of real production ML workflows in fraud detection:

| Experiment Step | Production Equivalent |
|----------------|----------------------|
| Velocity features (card in 1h window) | Real-time feature store (Redis/Flink) |
| Time-based CV split | Walk-forward validation on live traffic |
| Target encoding with OOF | Online label encoding with delay compensation |
| Threshold optimization | Precision/recall tradeoff tuning per business line |
| LightGBM + XGB ensemble | Model ensemble in risk scoring service |

The state machine architecture (external memory + deterministic loop) mirrors the MLOps pattern of experiment tracking (MLflow/W&B) + CI/CD for model retraining pipelines.
