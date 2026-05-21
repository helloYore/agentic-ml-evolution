# State Machine Prompt — V1 (House Prices)

Paste this prompt into Hermes to run a fully automated 10-round ML optimization loop.

---

## Prompt

```
任务：执行 Kaggle "house-prices-advanced-regression-techniques" 的 10 轮自动化迭代优化。

【底层约束】
1. 工具限制：仅使用 shell 交互（Kaggle CLI 数据下载与提交）、python_interpreter（代码执行）和 file_system（读写报告）。
2. 物理记忆外置：你当前的上下文随时可能被压缩。所有的分析思考、代码修改逻辑和实验结果，必须实时追加写入本地的 `evolution_report.md`。
3. 避免重复：每次修改代码前，必须先使用 shell 或文件工具读取 `evolution_report.md`，确认之前的迭代记录。

【执行状态机】（严格按序执行，当前初始化 N=1，上限 N=10）

状态 1 [Data & Baseline]（仅 N=1 时执行）:
- 使用 shell 下载并解压比赛数据。
- 编写基线模型（简单缺失值填充 + XGBoost 或 Random Forest），保存为本地 `train.py`。
- 执行脚本输出 `submission.csv`，并使用 Kaggle CLI 提交，附言 "Iteration 1 Baseline"。

状态 2 [Feedback & Record]:
- 强制执行命令 `sleep 60`。Kaggle 判题存在异步延迟，必须阻塞等待。
- 随后通过 Kaggle CLI 查询最新的 Public Score。
- 在 `evolution_report.md` 中追加写入《Iteration N 总结》，内容必须包含：本轮得分、特征工程策略、现有算法的缺陷分析。

状态 3 [Reflection & Evolution]（N < 10）:
- 基于上轮分析，提出 2-3 个具体的优化假设（例如：目标变量对数变换、处理离群点、引入多项式特征、尝试 LightGBM 等）。
- 将本次的优化思路写入 `evolution_report.md`。
- 重写 `train.py` 以实现这些优化。
- 生成新的 `submission.csv` 并提交。
- 内部计数器 N=N+1，跳回状态 2。

状态 4 [Final Synthesis]（N = 10 完成后执行）:
- 终止循环。读取完整的 `evolution_report.md`。
- 在文档末尾生成《全局进化总结报告》，对比基线与最终得分，量化分析哪些特征或算法优化对分数的提升最显著。
- 在终端输出 "10轮迭代闭环结束"。
```

---

## Design Notes

### Why External Memory?

LLM context windows get compressed as conversations grow. Without external state, the agent at iteration 7 has no reliable memory of what it tried at iteration 2. This causes two failure modes:

- **Regression**: Re-trying strategies that already failed
- **Redundancy**: Re-discovering improvements already captured

The `evolution_report.md` file acts as a **context-safe external memory** — it persists regardless of what happens to the LLM's internal context.

### Why a State Machine?

Free-form agent prompts drift. A state machine enforces a **read-before-write** contract:

```
State 3 rule: "每次修改代码前，必须先读取 evolution_report.md"
```

This single constraint prevents the majority of context-blindness failures.

### Replication

The prompt is fully self-contained. To replicate:
1. Install Hermes
2. Configure Kaggle credentials
3. Paste the prompt above
4. The agent does everything else — data download, modeling, submission, and reporting
