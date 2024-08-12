---
title: 因子投资理念
date: 2024-08-07 12:00:00 +0800
categories: [策略研究, 因子投资]
tags: [量化投资]
math: true
---

## 因子

因子是什么？用最简单的话来说：

**因子是资产收益的驱动力。**

从这个表述上来看，因子是一个抽象概念。从风险角度看：

**因子描述了众多资产共同暴露的系统性风险，因子所带来的收益正是这种系统性风险的风险溢价或风险补偿。**

> 风险换收益，想要获取什么样的收益，就要承担相应的风险。

## 多因子模型与异象

为进一步定量描述因子和资产收益的关系，有如下的多因子模型：

$$E(R_i^e) = \alpha_i + \beta_i^{\prime} \lambda$$

- 此处$E(R_i^e)$即为资产$i$的预期收益率，它来自于因子收益率$\lambda$。
- $\beta_i$因子收益率前的系数，表示资产对因子收益率的依赖程度，给它起一个装逼的名字——因子暴露。
- $\alpha_i$则是因子无法解释的收益率。

> 需要区分一件事：$\lambda$不是因子，某个变量（如ROE）不是因子，因子是一个抽象概念。

选定多因子模型后，若某个资产组合存在显著大于0的$\alpha$，则称这个资产组合为异象。

> 需要注意：异象指某个资产组合，而不是$\alpha$本身。$\alpha$可称为异象因子。

从有效市场假说出发，异象不应该存在。若发现真的存在，那就是多因子模型错了。因而学术界不断指出新的异象，又不断提出新的多因子模型，试图解释更多异象。

## 主要多因子模型

1. Fama and French(1993) 三因子模型

    资产收益来自市场因子(MKT)、价值因子(High Minus Low, HML)、规模因子(Small Minus Big, SMB)。

    $$E(R_i) - R_f = \beta_{i,MKT}(E(R_M)-R_f) + \beta_{i,SMB}E(R_{SMB}) + \beta_{i,HML}E(R_{i,HML})$$

    其观察到：低估值的股票相比于高估值的股票，未来收益率更高；规模较小的公司相比于规模较大的公司，未来收益率更高。

2. Carhart(1997) 四因子模型

    加入了截面动量因子(MOM)。

    $$E(R_i) - R_f = \beta_{i,MKT}(E(R_M)-R_f) + \beta_{i,SMB}E(R_{SMB}) + \beta_{i,HML}E(R_{i,HML}) + \beta_{i,MOM}E(R_{i,MOM})$$

    其观察到：在美股中，过去涨得好的股票，在未来也更可能涨得好。

3. Novy-Marx(2013) 四因子模型

    加入了盈利因子(Profitability Minus Unrofitability, PMU)。

    $$E(R_i) - R_f = \beta_{i,MKT}(E(R_M)-R_f) + \beta_{i,SMB}E(R_{SMB}) + \beta_{i,PMU}E(R_{i,PMU}) + \beta_{i,UMD}E(R_{i,UMD})$$

    其观察到，公司的盈利能力与未来收益密切相关。

4. Fama and French(2015) 五因子模型

    相比于三因子模型，加入了盈利因子(RMW)和投资因子(CMA)。

    $$E(R_i) - R_f = \beta_{i,MKT}(E(R_M)-R_f) + \beta_{i,SMB}E(R_{SMB}) + \beta_{i,HML}E(R_{i,HML}) + \beta_{i,RMW}E(R_{i,RMW}) + \beta_{i,CMA}E(R_{i,CMA})$$

5. Hou-Xue-Zhang(2015) 四因子模型

    从实体投资经济学出发，得到规模因子、投资因子、盈利因子。

    $$E(R_i) - R_f = \beta_{i,MKT}(E(R_M)-R_f) + \beta_{i,ME}E(R_{ME}) + \beta_{i,I/A}E(R_{i,I/A}) + \beta_{i,ROE}E(R_{i,ROE})$$

6. Stamnaugh-Yuan(2017) 四因子模型

    加入管理因子和表现因子

7. Daniel-Hirshleifer-Sun(2020) 三因子模型

    用行为金融学理论构建因子。