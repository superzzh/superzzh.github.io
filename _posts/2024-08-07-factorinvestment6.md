---
title: 投资组合优化：均值-方差模型
date: 2024-08-07 12:00:00 +0800
categories: [策略研究, 因子投资]
tags: [量化投资]
math: true
---

结合收益模型和风险模型做投资组合优化，期望在控制风险的同时，取得尽可能高的收益。

## Markowitz均值-方差模型

求解目标：

$$\underset {w}{min} \quad R^{\prime}w-\frac{\zeta}{2}w^{\prime} \Sigma w$$

一方面，它想要找到股票组合权重向量$w$，使得组合收益$R^{\prime}w$尽可能大。另一方面，又对风险$w^{\prime} \Sigma w$做了惩罚。

