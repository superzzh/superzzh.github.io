---
title: 风险模型：Barra模型
date: 2024-08-07 12:00:00 +0800
categories: [策略研究, 因子投资]
tags: [量化投资]
math: true
---

## 为什么协方差矩阵代表了风险？

对于投资组合$wR$来说，其波动率可用标准差来衡量。为此，需要得到方差，而资产组合的方差用协方差矩阵表达。

## 风险模型的目的

风险模型的关键在于资产收益协方差矩阵的估计。而多因子模型的一个重要意义，是实现了降维。

假设共有$N$个资产，取时间段为$T$，多因子模型中包含$K$个因子，则有收益率矩阵：

$$\pmb{R}=[R_1,R_2,...,R_N]^{\prime}=\left[\begin{matrix}
R_{11} & R_{12} & ... & R_{1T} \\
R_{21} & R_{22} & ... & R_{2T} \\
...  & ...  & ... & ...  \\
R_{N1} & R_{N2} & ... & R_{NT} \\
\end{matrix}\right]_{(N \times T)}$$

有因子暴露矩阵，注意$\beta_{ij}$表示资产$j$在因子$i$上的暴露：

$$\pmb{\beta} = [\beta_1, \beta_2, ..., \beta_K]=\left[\begin{matrix}
\beta_{11} & \beta_{21} & ... & \beta_{K1} \\
\beta_{12} & \beta_{22} & ... & \beta_{K2} \\
...  & ...  & ... & ...  \\
\beta_{1N} & \beta_{2N} & ... & \beta_{KN} \\
\end{matrix}\right]_{(N \times K)}$$

因子收益率时间序列构成矩阵，注意$\lambda_{ij}$表示因子$i$在$j$期的取值：

$$\pmb{\lambda} = [\lambda_1, \lambda_2, ..., \lambda_K]^{\prime}=\left[\begin{matrix}
\lambda_{11} & \lambda_{12} & ... & \lambda_{1T} \\
\lambda_{21} & \lambda_{22} & ... & \lambda_{2T} \\
...  & ...  & ... & ...  \\
\lambda_{K1} & \lambda_{K2} & ... & \lambda_{KT} \\
\end{matrix}\right]_{(K \times T)}$$

类似的，有特质收益率矩阵：

$$\pmb{\epsilon} = [\epsilon_1, \epsilon_2, ..., \epsilon_N]^{\prime}=\left[\begin{matrix}
\epsilon_{11} & \epsilon_{12} & ... & \epsilon_{1T} \\
\epsilon_{21} & \epsilon_{22} & ... & \epsilon_{2T} \\
...  & ...  & ... & ...  \\
\epsilon_{N1} & \epsilon_{N2} & ... & \epsilon_{NT} \\
\end{matrix}\right]_{(N \times T)}$$

则有：

$$\pmb{R = \beta \lambda + \epsilon}$$

在三条假设：

- 因子收益率经过标准化
- 因子收益率与特质收益率无关，即$E(\pmb{\epsilon\lambda})=0$
- 不同期的因子收益率之间没有相关性，即$E(\epsilon_i \epsilon_j^{\prime})=0$

求协方差矩阵可得：

$$\Sigma = \beta \Sigma_f \beta^{\prime} + \Sigma_{\epsilon}$$

资产收益率协方差矩阵本为$N$阶矩阵，当资产数量过多时，这个矩阵维度非常高，不好使用。但是，若能找到一些因子来解释资产收益的来源，就可以用低维$K$阶矩阵$\Sigma_f$（$K<N$）来代替高维矩阵$\Sigma$。SURPRISE!实现降维。

多因子模型有**两个视角，两方面的作用**：

- 收益：用因子收益解释资产预期收益的截面差异
- 风险：准确地估计协方差矩阵

## Barra CNE5多因子模型

Barra CNE5多因子模型在风格因子之外，还加入了1个国家因子（本人理解为市场因子）、$P$个行业因子，再加上$Q$个风格因子，共$K=1+P+Q$个因子。

在t期，该多因子模型为：

$$\left[\begin{matrix}
R_{1t}^e  \\
R_{2t}^e  \\
...    \\
R_{Nt}^e  \\
\end{matrix}\right]=\left[\begin{matrix}
1  \\
1  \\
...    \\
1  \\
\end{matrix}\right]\lambda_{Ct}+\left[\begin{matrix}
\beta_{1,t-1}^{I_1}  \\
\beta_{2,t-1}^{I_1}  \\
...    \\
\beta_{N,t-1}^{I_1}  \\
\end{matrix}\right]\lambda_{I_1t}+...+\left[\begin{matrix}
\beta_{1,t-1}^{I_P}  \\
\beta_{2,t-1}^{I_P}  \\
...    \\
\beta_{N,t-1}^{I_P}  \\
\end{matrix}\right]\lambda_{I_Pt}+\left[\begin{matrix}
\beta_{1,t-1}^{S_1}  \\
\beta_{2,t-1}^{S_1}  \\
...    \\
\beta_{N,t-1}^{S_1}  \\
\end{matrix}\right]\lambda_{S_1t}+...+\left[\begin{matrix}
\beta_{1,t-1}^{S_Q}  \\
\beta_{2,t-1}^{S_Q}  \\
...    \\
\beta_{N,t-1}^{S_Q}  \\
\end{matrix}\right]\lambda_{S_Qt}+\left[\begin{matrix}
\epsilon_{1t}  \\
\epsilon_{2t}  \\
...    \\
\epsilon_{Nt}  \\
\end{matrix}\right]$$

该模型有三个特点：

1. 假设股票在t期的收益率和因子在t-1期的暴露是已知的，且因子暴露直接用公司的特征变量值代替。则模型求解的目的是因子收益率。

2. 每个公司在一期内只能属于一个行业，故有：

    $$\beta_{i,t-1}^{I_1}+\beta_{i,t-1}^{I_2}+...+\beta_{i,t-1}^{I_P}=1, \quad i=1,2,...,N$$

    但是，这样会造成共线性，导致模型出现多个解（不满秩）。于是要做约束：设市值权重向量为$w$，则需要满足：

    $$
    \begin{equation} 
    [w_1,w_2,...,w_N]\left[\begin{matrix}
    \beta_{1,t-1}^{I_1} & \beta_{2,t-1}^{I_1} & ... & \beta_{N,t-1}^{I_1} \\
    \beta_{1,t-1}^{I_2} & \beta_{2,t-1}^{I_2} & ... & \beta_{N,t-1}^{I_2} \\
    ...  & ...  & ... & ...  \\
    \beta_{1,t-1}^{I_P} & \beta_{2,t-1}^{I_P} & ... & \beta_{N,t-1}^{I_P} \\
    \end{matrix}\right]\left[\begin{matrix}
    \lambda_{I_1t}  \\
    \lambda_{I_2t}  \\
    ...    \\
    \lambda_{I_P-t}  \\
    \end{matrix}\right]=\left[\begin{matrix}
    0  \\
    0  \\
    ...    \\
    0  \\
    \end{matrix}\right]
    \end{equation}
    $$

    记$s_{I_p}$为所有属于行业$I_p$的股票按照市值计算出的权重之和：

    $$s_{I_p} = \sum_{i=1}^{N}w_i\beta_{i,t-1}^{I_p}$$

    则上面的式子可以简写为:

    $$s_{I_1}\lambda_{I_1}+s_{I_2}\lambda_{I_2}+...+s_{I_P}\lambda_{I_P}=0$$

3. 在因子暴露值的预处理上也有讲究。对于任意风格因子$S_q$，其因子暴露为$\beta_{i,t-1}^{S_q}$，设市值权重向量为$w$：

    - 去均值：使用因子暴露原始值减去因子暴露的市值加权平均值。

        $$\bar{\beta}_{t-1}^{S_q} = \sum_{i=1}^{N}w_i\beta_{i,t-1}^{S_q}$$

    - 除以标准差：使用去均值后的变量除以标准差。

        $$\sigma(\beta_{i,t-1}^{S_q})=\sqrt{\frac{1}{N}\sum_{i=1}^{N}\beta_{i,t-1}^{S_q}}$$

        $$\tilde{\beta}_{i,t-1}^{S_q} = \frac{\beta_{i,t-1}^{S_q}-\bar{\beta}_{t-1}^{S_q}}{\sigma(\beta_{i,t-1}^{S_q})}$$
    
    为什么要这样？这样操作有什么好处？

    考虑市值加权的股票组合，有：

    $$
    \begin{equation}
    \begin{aligned}
        wR_t &= w_1R_{1t}+w_2R_{2t}+...+w_NR_{Nt} \\
        &=\lambda_{Ct}+\sum_{p=1}^{P}s_{I_p}\lambda_{I_pt}+
        \sum_{q=1}^{Q} (\sum_{i=1}^{N}w_i\tilde{\beta}_{i,t-1}^{S_q}) \lambda_{S_qt}+
        \sum_{i=1}^{N}w_{i}\epsilon_{it}
    \end{aligned}
    \end{equation}
    $$

    观察：

    $$\sum_{i=1}^{N}w_i\tilde{\beta}_{i,t-1}^{S_q}=\frac{\sum_{i=1}^{N}w_i\beta_{i,t-1}^{S_q} - \sum_{i=1}^{N}w_i\bar{\beta}_{t-1}^{S_q}}{\sigma(\beta_{i,t-1}^{S_q})}=0$$

    又有：

    $$\sum_{i=1}^{N}w_{i}\epsilon_{it} \approx 0$$

    故：

    $$wR_t \approx \lambda_{Ct}$$

    这说明：**市值加权组合约等于国家因子纯因子组合**。

## 模型求解

### 异方差与加权最小二乘法

模型出现异方差时，有：

$$Var(\pmb{\epsilon}|X) = \sigma^2\left[\begin{matrix}
w_1 &  &  &  \\
 & w_2 &  &  \\
  &   & ... &   \\
 &  &  & w_n \\
\end{matrix}\right] = \sigma^2 \Omega$$

令：

$$W=\Omega^{-1}=\left[\begin{matrix}
    \frac{1}{w_1} &  &  &  \\
    & \frac{1}{w_2} &  &  \\
    &   & ... &   \\
    &  &  & \frac{1}{w_n} \\
    \end{matrix}\right]=\left[\begin{matrix}
    \frac{1}{\sqrt{w_1}} &  &  &  \\
    & \frac{1}{\sqrt{w_2}} &  &  \\
    &   & ... &   \\
    &  &  & \frac{1}{\sqrt{w_n}} \\
    \end{matrix}\right]^2 = 
    P^{\prime}P$$

则求解加权最小二乘法：$PY=PX\beta+P\epsilon$，得

$$\begin{aligned}
\hat{\beta} &= (X^{\prime}P^{\prime}PX)^{-1}X^{\prime}P^{\prime}PY \\
            &= (X^{\prime}WX)^{-1}X^{\prime}WY
\end{aligned}$$

### 求解条件设定

采用了如下回归权重矩阵：

$$W=\left[\begin{matrix}
\frac{\sqrt{s_1}}{\sum_{i=1}^{N}\sqrt{s_i}} &  &  &  \\
 & \frac{\sqrt{s_2}}{\sum_{i=1}^{N}\sqrt{s_i}} &  &  \\
  &   & ... &   \\
 &  &  & \frac{\sqrt{s_N}}{\sum_{i=1}^{N}\sqrt{s_i}} \\
\end{matrix}\right]$$

约束条件中，$\lambda_{I_P}$可以写成其他$\lambda_{I_p}$的线性组合，如下形式：

$$
    \begin{equation} 

    \left[\begin{matrix}
    \lambda_{Ct} \\
    \lambda_{I_1} \\
    \lambda_{I_2} \\
    \vdots \\
    \lambda_{I_P} \\
    \lambda_{S_1} \\
    \lambda_{S_2} \\
    \vdots\\
    \lambda_{S_Q} \\
    \end{matrix}\right]
    =
    \left[\begin{matrix}
    1 & 0 & ... & 0 & 0\\
    0 & 1 & ... & 0 & 0\\
    0 & 1 & ... & 0 & 0\\
    ...  & ...  & ... & ... & ...\\
    -\frac{s_{I_1}}{s_{I_P}} & -\frac{s_{I_2}}{s_{I_P}} & ... & -\frac{s_{I_{P-2}}}{s_{I_P}} & -\frac{s_{I_{P-1}}}{s_{I_P}}\\
    0 & 0 & ... & 0 & 0\\
    0 & 0 & ... & 1 & 0 \\
    0 & 0 & ... & 0 & 1 \\
    \end{matrix}\right]
    
    \left[\begin{matrix}
    \lambda_{Ct} \\
    \lambda_{I_1} \\
    \lambda_{I_2} \\
    \vdots \\
    \lambda_{I_{P-1}} \\
    \lambda_{S_1} \\
    \lambda_{S_2} \\
    \vdots\\
    \lambda_{S_Q} \\
    \end{matrix}\right]
    
    \end{equation}
$$

记等号右边的矩阵为$C$，其是$K \times (K-1)$维矩阵，带约束的加权最小二乘情况下，纯因子投资组合权重矩阵为：

$$\Omega = C(C^{\prime} \beta^{\prime} W \beta C)^{-1} C^{\prime} \beta^{\prime} W$$

验证一下易知：$\Omega \beta = I$

### 因子收益率

在得到$\Omega$之后，易得每个因子t期的收益率：

$$\lambda_{kt} = \sum_{i=1}^{N} \omega_{ki}R_{it}^{e}, \quad k=1,2,...,K$$

式中$\omega_{ki}$是纯因子权重矩阵第k行第i个元素。可求得第k个因子的因子收益率。

## 风格因子体系

## 严格控制因子暴露

市值敞口

行业中性/行业偏离

个股集中度


