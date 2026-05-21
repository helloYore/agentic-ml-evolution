# State Machine Prompt — V2 (Fraud Detection)

Adapted from V1 for financial fraud detection. Key changes: AUC-ROC metric, time-based CV, imbalanced classification, velocity feature engineering.

---

## Prompt

```
任务：执行 Kaggle "ieee-fraud-detection" 的 N 轮自动化迭代优化（上限20轮，满足收敛条件可提前终止）。

【底层约束】
1. 工具限制：仅使用 shell、python_interpreter、file_system。
2. 物理记忆外置：所有分析、代码修改逻辑、实验结果，实时追加写入 evolution_report.md。
3. 避免重复：每次修改代码前，必须先读取 evolution_report.md 确认历史记录。
4. 收敛终止：连续5轮 CV AUC 未提升（持平不算提升），自动终止。

【状态机】（严格按序执行，初始化 N=1）

状态 0 [EDA & Setup]（仅 N=1 时执行一次）:
- 下载并解压数据：
  kaggle competitions download -c ieee-fraud-detection
- 编写并执行 eda.py，输出以下分析：
  - 类别不平衡比例（fraud rate）
  - TransactionDT 范围 → 推导 hour_of_day / day_of_week / days_since_start
  - TransactionAmt 分布（均值/中位数/99th percentile）
  - identity 表 join 覆盖率（有多少 transaction 有 identity 记录）
  - 缺失值 Top 20 特征
  - 各类别特征（ProductCD / card4 / card6 / P_emaildomain）的欺诈率
- 将上述分析固化为《全局数据字典》写入 evolution_report.md。
- 初始化 N=1，进入状态 1。

状态 1 [Model & Execute]:
- N=1：基于数据字典，编写基线 train.py:
  - LightGBMClassifier(n_estimators=500, lr=0.05, class_weight='balanced')
  - 时序验证（前80% TransactionDT 训练，后20%验证，禁止随机split）
  - 仅使用原始特征，不做额外特征工程
  - 输出：CV AUC + F1 @ optimal threshold
- N>1：基于状态3的决策重写 train.py。
- 执行并打印：
  Train AUC: x.xxxxx | Val AUC: x.xxxxx | F1@thresh: x.xxx (threshold=x.xx)
  Top 10 features: [feature: importance, ...]
  Fraud sample residuals: [最大误判的10个 TransactionID]

状态 2 [Record & Diagnosis]:
- 将本轮结果追加写入 evolution_report.md：
  ## Iteration N — <strategy-summary>
  ### 策略
  <本轮相较上轮的具体变更>
  ### 结果
  - Val AUC: x.xxxxx (Δ vs prev: ±x.xxxxx)
  - F1 @ threshold: x.xxx
  - Top features: [...]
  ### 缺陷分析
  1. <issue 1>
  2. <issue 2>
- 如 Val AUC 是历史最优，记录 best_iter=N。
- 检查收敛条件：连续5轮未提升 → 跳转状态4。

状态 3 [Reflection & Evolution]（N < max_rounds）:
- 读取 evolution_report.md 最近2轮 + 全局数据字典。
- 基于残差分析（最大误判样本）提出2-3个优化假设，格式：
  假设 H1: [具体操作]
  理由: [为什么可能有效]
  风险: [可能的副作用]
- 重点探索方向（按预期收益排序）：
  1. 速度特征：card1/card2 在 1h/1d/7d 窗口内的交易次数/金额均值/标准差
  2. 时间特征：hour_of_day / day_of_week / is_weekend / days_since_start
  3. identity 表 join 及设备特征编码
  4. 高基数类别特征目标编码（OOF 方式，避免泄漏）
  5. scale_pos_weight 调参 vs SMOTE 对比
  6. XGBoost 对比验证
  7. LightGBM + XGBoost 加权集成
- 将优化假设写入 evolution_report.md，重写 train.py，N+=1，返回状态1。

状态 4 [Final Synthesis]（收敛或 N=max_rounds 后执行）:
- 读取完整 evolution_report.md。
- 生成《全局进化总结》：
  - AUC 轨迹表
  - 最有效特征组（velocity/time/identity/encoding）的量化贡献
  - 无效策略清单
  - 最优模型对应的 Iter N，生成最终 submission.csv 并提交
- 输出："N轮迭代闭环结束，最优 CV AUC=x.xxxxx"
```

---

## Key Differences from V1

### Validation Strategy

```python
# V1: random 5-fold (acceptable for i.i.d. regression)
from sklearn.model_selection import KFold

# V2: time-based split (mandatory for fraud — prevents leakage)
# Fraud models trained on future data to predict past = optimistic bias
train_idx = df['TransactionDT'] < df['TransactionDT'].quantile(0.8)
val_idx = df['TransactionDT'] >= df['TransactionDT'].quantile(0.8)
```

### Velocity Feature Template

```python
# Example: card1-level velocity features
# Must be computed WITHOUT leaking future information
df = df.sort_values('TransactionDT')

for window_hours in [1, 24, 168]:  # 1h, 1d, 7d
    window_rows = window_hours * 3600  # TransactionDT is in seconds
    df[f'card1_count_{window_hours}h'] = (
        df.groupby('card1')['TransactionDT']
        .transform(lambda x: x.expanding().count())  # simplified; use rolling in practice
    )
    df[f'card1_amt_mean_{window_hours}h'] = (
        df.groupby('card1')['TransactionAmt']
        .transform('mean')
    )
```

### Imbalance Handling

```python
# Option A: class_weight (simplest, usually best starting point)
model = lgb.LGBMClassifier(class_weight='balanced')

# Option B: scale_pos_weight (explicit ratio)
fraud_ratio = y_train.sum() / len(y_train)
model = lgb.LGBMClassifier(scale_pos_weight=(1 - fraud_ratio) / fraud_ratio)

# Option C: SMOTE (often over-hyped for tabular fraud data — test carefully)
from imblearn.over_sampling import SMOTE
X_res, y_res = SMOTE(random_state=42).fit_resample(X_train, y_train)
```

### Threshold Optimization

```python
from sklearn.metrics import f1_score
import numpy as np

# AUC is the Kaggle metric, but real deployment needs a threshold
thresholds = np.arange(0.1, 0.9, 0.01)
f1_scores = [f1_score(y_val, (y_prob > t).astype(int)) for t in thresholds]
best_threshold = thresholds[np.argmax(f1_scores)]
```
