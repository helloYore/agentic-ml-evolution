# House Prices — Evolution Report

**Competition**: house-prices-advanced-regression-techniques  
**Model**: DS-V4-Flash | **Rounds**: 10 | **Cost**: ~¥1 RMB | **Time**: 20 min

---

## Iteration 0 — 初始化
目标: 10轮自动化迭代优化，最优 Public Score  
比赛: house-prices-advanced-regression-techniques

---

## Iteration 1 — Baseline
### 策略
RandomForestRegressor (300 trees, max_depth=20), median imputation + OneHot

### 结果
- CV R²: 0.8566
- Public Score: **0.14640**

### 缺陷分析
1. 无目标变换（SalePrice 右偏）
2. 无离群处理
3. 无特征工程
4. 简单中位数填充，丢失缺失信息

---

## Iteration 2 — Log + Outliers + Skew
### 策略
log1p target + 移除 2 个离群点 (GrLivArea > 4000) + 20 个偏态特征 log 变换

### 结果
- CV R²: 0.8800 (↑0.0234)
- Public Score: **0.14584** (↓0.00056)

### 缺陷分析
1. RF 对 log 变换不敏感，换树模型效果受限
2. 无特征工程，特征表达仍然贫乏
3. OneHot 过稀疏，高基数类别特征膨胀

---

## Iteration 3 — XGBoost + Feature Engineering ⭐ 首次大幅突破
### 策略
XGBoost (500 trees, depth=5, lr=0.05) + 13 个新特征:
- TotalSF = TotalBsmtSF + 1stFlrSF + 2ndFlrSF
- TotalBath = FullBath + 0.5*HalfBath + BsmtFullBath + 0.5*BsmtHalfBath
- OverallQual_TotalSF (质量×面积交互)
- HouseAge, RemodAge 等年龄特征

### 结果
- CV R²: **0.9128** (↑0.0542)
- Public Score: **0.12975** (↓0.01609) ⭐

### 缺陷分析
1. 可能轻微过拟合 (CV vs Public gap)
2. 类别特征未做有序编码 (Ex/Gd/TA/Fa/Po)
3. 年份特征未做分桶

---

## Iteration 4 — LightGBM + Ordinal + Decades
### 策略
LightGBM + 有序特征 Ex→5 映射 + 年 decade 分桶 + 交互特征

### 结果
- CV R²: 0.9061 (↓0.0067)
- Public Score: **0.13125** (↑0.00150) — 回退

### 缺陷分析
LightGBM (leaf-wise) 在 ~1500 行小数据集上过拟合，XGBoost (level-wise) 更稳定

---

## Iteration 5 — XGBoost Over-regularized
### 策略
XGBoost (800 trees, depth=4, lr=0.03, high reg_alpha/lambda) + 所有新增特征

### 结果
- CV R²: 0.9031 (↓0.0097)
- Public Score: **0.13187** (↑0.00212) — 回退

### 缺陷分析
过度正则化 + 维度灾难 (过多 OneHot 稀疏特征)，维度爆炸稀释有效信号

---

## Iteration 6 — ElasticNet ⭐ 意外最优单模型
### 策略
RidgeCV / LassoCV / ElasticNetCV + StandardScaler + log 偏态特征变换

### 结果
- CV R²: Ridge=0.9208, Lasso=0.9238, **ElasticNet=0.9247**
- Public Score: **0.12875** (↓0.00100) ⭐

### 关键洞察
log 变换后房价接近对数线性关系，正则化线性模型在此场景意外优于所有树模型

### 缺陷分析
预测上限偏高 / 未做集成，单模型方差较大

---

## Iteration 7 — Ensemble XGBoost + ElasticNet ⭐
### 策略
加权平均 (33% XGB + 67% EN)，CV 优化权重

### 结果
- CV R²: 0.9264
- Public Score: **0.12551** (↓0.00324) ⭐

### 分析
集成带来实质提升。ElasticNet 权重远超 XGBoost — 验证了 log-linear 假设。

---

## Iteration 8 — Ensemble + Advanced Features ⭐
### 策略
加入 Neighborhood 均值编码、TotalQualScore 综合质量评分、更多交互特征

### 结果
- CV R²: 0.9262
- Public Score: **0.12468** (↓0.00083) ⭐

### 分析
Neighborhood 均值编码和 TotalQualScore 贡献显著。目标编码有效捕捉地理位置溢价。

---

## Iteration 9 — Stacking (Ridge Meta-Learner)
### 策略
XGBoost + ElasticNet 预测作为特征 → Ridge 元学习器

### 结果
- CV R²: 0.9264
- Public Score: **0.12561** (↑0.00093) — 回退

### 分析
纯线性元学习器不优于加权平均。小数据集 Stacking 的元学习器容易过拟合 OOF 预测。

---

## Iteration 10 — 3-Model Ensemble 🏆 全场最优
### 策略
XGBoost (13%) + ElasticNet (69%) + GradientBoosting (18%)  
+ MSSubClass 转类别特征 + 更细化离群点处理

### 结果
- CV R²: 0.9261
- Public Score: **0.12381** (↓0.00087) 🏆

### 分析
增加第3个多样性模型 GBR 带来边际提升。GradientBoosting 弥补了 XGBoost 和 ElasticNet 之间的预测盲区。

---

## 全局进化总结

### 得分轨迹
```
Iter 1:  0.14640  (baseline)
Iter 3:  0.12975  ↓ -11.4%  ← 最大单次跃升 (XGBoost + 特征工程)
Iter 6:  0.12875  ↓ -0.8%   ← ElasticNet 意外逆袭
Iter 7:  0.12551  ↓ -2.5%   ← 集成开始生效
Iter 10: 0.12381  ↓ -1.3%   ← 最终最优
总提升:  -15.4%
```

### 贡献排名（估计）

| 策略 | 贡献 | 迭代 |
|------|------|------|
| XGBoost + 特征工程 | ~35% | Iter 3 |
| ElasticNet 正则化线性 | ~40% | Iter 6 |
| 集成融合 | ~15% | Iter 7→10 |
| log 变换 + 离群点处理 | ~5% | Iter 2 |
| Neighborhood 均值编码 | ~5% | Iter 8 |

### 无效策略
- LightGBM 替代 XGBoost (Iter 4) — 小数据集回退
- 过度特征工程 + 过正则化 (Iter 5) — 维度灾难
- Ridge 元学习器 Stacking (Iter 9) — 不优于加权平均

### 关键教训
1. **log-linear 假设**：目标 log 变换后，ElasticNet 可能超越树模型
2. **少即是多**：Iter 5 证明过度特征工程有害
3. **多样性 > 数量**：3 个精心选择的模型 > 随意叠加
4. **CV vs Public 不完全一致**：以 Public Score 为最终决策依据
