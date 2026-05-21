# State Machine Prompt — V4 Advanced (House Prices)

Upgraded from V1. Key improvements: EDA-first state, CV Std stability tracking, convergence termination, structured hypothesis format, no per-round Kaggle submission (cost-efficient).

---

## What Changed vs V1

| Dimension | V1 (Basic) | V4 (Advanced) |
|-----------|-----------|---------------|
| Rounds | Fixed 10 | Up to 50, early-stop on convergence |
| Kaggle submission | Every round | Final best only |
| Primary metric | Public Score | 5-Fold CV RMSLE |
| EDA | None | Dedicated State 0, builds data dictionary |
| Stability tracking | None | CV Std monitored each round |
| Convergence | None | Auto-stop after 5 rounds without improvement |
| Hypothesis format | Free-form | Structured (EDA + H + rationale + risk) |

---

## Prompt

```
任务：从零开始，探索 Kaggle House Prices 数据集上当前 pipeline 能达到的最优 CV RMSLE。
轮次上限为50轮，满足收敛条件可提前终止。

【底层约束】
1. 工具限制：仅使用 shell、python_interpreter、file_system。
2. 评价准则：唯一指标是 5-Fold CV RMSLE Mean。不提交到 Kaggle 直到最终最优模型。
3. 实验追踪：
   - 报告文件：evolution_report.md（静态，追加写入）
   - 每次修改 train.py 前，先读取报告的最近 2 轮 + 全局数据字典，避免重复造轮
4. 依赖管理：需要时自动 pip install。
5. 防过拟合：
   - 每轮记录 CV Mean + CV Std
   - 连续 5 轮 CV 没有改善（持平不算改善），触发收敛终止
   - 任何导致 CV Std 比上轮增大超过 50% 的变化，标记为"不稳定"，除非后续 2 轮验证其有效性

【执行状态机】（严格按序执行，初始化 N=1）

状态 0 [Pre-EDA & Setup]（仅执行一次）:
1. 编写并执行 pre_eda.py，进行全面 EDA：
   - 缺失值统计（合理缺失 vs 真实缺失）
   - 数值特征偏度分析
   - 目标变量分布（SalePrice skew，log1p 后 skew）
   - 类别特征基数统计
   - 离群点检测（GrLivArea vs SalePrice 散点图，统计 IQR）
2. 将 EDA 结果固化为《全局数据字典》写入 evolution_report.md。
3. N=1，进入状态 1。

状态 1 [Model & Execute]:
- N=1：基于数据字典，编写基线 train.py（基本清洗 + 随机森林回归）。
- N>1：基于状态 3 的决策重写 train.py。
- 执行并打印：
  Fold 1: 0.xxxxxx
  Fold 2: 0.xxxxxx
  ...
  CV RMSLE Mean: 0.xxxxxx
  CV RMSLE Std:  0.xxxxxx
  同时内联输出：
  - 特征重要性 Top 10
  - 按价格5分位的平均残差（验证回归均值效应程度）
  - 残差最大的 20 个样本 ID + 误差率

状态 2 [Record & Diagnosis]:
1. 将本轮结果追加写入 evolution_report.md 的《Iteration N 总结》：
   - 模型/特征变更摘要
   - CV Mean / Std
   - 特征重要性 Top 5 + Bottom 5
   - 价格分位残差表
   - 结论（有效 / 无效 / 不稳定）
2. 如果 CV Mean 是历史最优，记录 best_iter=N。
3. 检查收敛条件：连续 5 轮 CV Mean 未改善 → 跳转状态 4。

状态 3 [Reflection & Evolution]（N < 50）:
- 读取 evolution_report.md 最近 2 轮 + 全局数据字典。
- 基于上一轮 bad case 做针对性 EDA，然后提出 2-3 个优化假设，格式：
  EDA结论: [基于上一轮 bad case 做的 EDA]
  假设 H1: [一句话描述]
  理由: [为什么这个假设可能有效]
  风险: [可能过拟合/退步的原因]
- 将假设写入 evolution_report.md，重写 train.py，N+=1，返回状态 1。

状态 4 [Final Synthesis]（收敛或 N=50 后执行）:
1. 读取完整 evolution_report.md，生成《全局进化总结》。
2. 确保 submission.csv 使用的是 best_iter 对应的最优模型。
3. 使用 Kaggle CLI 提交最终结果。
4. 输出："V4 自动化迭代闭环结束，最优 CV=xxx"
```

---

## Design Notes: Why V4 Is Better

### EDA-First (State 0)
V1 jumps straight to modeling. V4 builds a **data dictionary** first — a fixed reference for every subsequent iteration. This prevents the agent from "re-discovering" the same data facts across rounds and focuses hypotheses on what actually matters.

### CV Std as Stability Signal
Tracking CV Mean alone misses overfitting. A model with CV Mean=0.120 but Std=0.015 is less reliable than CV Mean=0.122, Std=0.005. The Std gate catches unstable improvements before they compound.

### No Per-Round Submission
V1 submits to Kaggle every round. At 10+ rounds, that's 10 API calls and score queries. V4 uses local CV as the primary signal and submits only once at the end — faster, cheaper, and forces better offline evaluation discipline.

### Structured Hypothesis Format
Free-form hypotheses drift. The `EDA结论 / 假设 / 理由 / 风险` format forces the agent to ground each proposal in observed data, reducing hallucinated or redundant strategies.
