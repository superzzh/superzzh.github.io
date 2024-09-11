---
title: 主流因子解读
date: 2024-08-07 12:00:00 +0800
categories: [策略研究, 因子投资]
tags: [量化投资]
math: true
---

以下因子均为风格因子，用排序法检验。为了统一效果，对检验设置做如下规定：

- 股票池：主板+中小板股票，科创板与创业板不纳入。
- 检验时间段：2013年1月1日至2023年12月31日。
- 分组：10组。
- 调仓周期：22日/月末调仓。
- 回测资金：100万元。

## 工具函数

```python
import rqdatac
import rqfactor
from rqfactor import Factor, execute_factor, FactorAnalysisEngine, Winzorization, QuantileReturnAnalysis, ICAnalysis
from rqfactor.extension import UserDefinedLeafFactor
from rqfactor.extension import rolling_window
from rqfactor.extension import RollingWindowFactor

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import calendar 
import datetime
```


```python
rqdatac.init()
```

```python
# 工具函数

## 股票池：主板+中小板
def stock_filter(dataframe):
    query = (dataframe['special_type']=='Normal') & \
             (dataframe['order_book_id'].str.startswith('000') | dataframe['order_book_id'].str.startswith('001') \
            | dataframe['order_book_id'].str.startswith('002') | dataframe['order_book_id'].str.startswith('003') \
            | dataframe['order_book_id'].str.startswith('600') | dataframe['order_book_id'].str.startswith('601') \
            | dataframe['order_book_id'].str.startswith('603'))
    return query

## 按照市值加权，获取权重向量
def get_weight_by_market_cap(ids, date):
    market_cap = rqdatac.get_factor(ids, 'market_cap', date, date)
    market_cap['market_cap'] = market_cap['market_cap'].fillna(0)
    weight = market_cap['market_cap']/sum(market_cap['market_cap'])
    return weight.values

## 按照市值加权，获取股票组合
def get_target_stock_sub_df(day):
    all_stock_df = rqdatac.all_instruments(type='CS', date=day)
    query = stock_filter(all_stock_df)
    target_stock_df = all_stock_df[query]

    target_stock_list = target_stock_df['order_book_id'].values
    date_list = [day for _ in range(len(target_stock_list))]
    weight = get_weight_by_market_cap(target_stock_list, day)

    sub_df = pd.DataFrame(data={'TRADE_DT':date_list, 'TICKER':target_stock_list, 'TARGET_WEIGHT':weight})
    return sub_df

## 双重独立检验
def Double_Factor_Analysis(book_ids, factor1, factor2, f1, f2, start_date, end_date, period):
    '''
    双因子检验
    目前两个因子都是从大到小排列
    '''
    factor_data = rqdatac.get_factor(ids, [factor1, factor2], start_date, end_date, expect_df=False)
    factor_data_group = factor_data.groupby('date') 
    close_data = rqfactor.execute_factor(Factor('close'), ids, start_date, end_date)
    returns_data = close_data.pct_change()

    title_list = []
    for i in range(f1):
        for j in range(f2):
            title = f'Factor1.{i+1}~Factor2.{j+1}'
            title_list.append(title)

    trade_days = pd.to_datetime(rqdatac.get_trading_dates(start_date, end_date, market='cn'))

    portfolio_dict = {title:[] for title in title_list}
    portfolio_daily_returns = pd.DataFrame(data=None, columns=title_list, index=trade_days[1:])
    
    flag = 0
    for day in range(len(trade_days)-1):

        trade_day = trade_days[day]
        target_day = trade_days[day+1]

        if flag % period == 0:
            transfer = True
            flag = 0

        df = factor_data_group.get_group(trade_day)
        df = df.dropna()

        for i in range(f1):
            for j in range(f2):
                title = f'Factor1.{i+1}~Factor2.{j+1}'
                if transfer:
                    query_factor1 = (df[factor1] >= np.quantile(df[factor1],i/f1)) & (df[factor1] < np.quantile(df[factor1],(i+1)/f1))
                    query_factor2 = (df[factor2] >= np.quantile(df[factor2],j/f2)) & (df[factor2] < np.quantile(df[factor2],(j+1)/f2))
                    stock_list = df[query_factor1 & query_factor2].index.get_level_values('order_book_id').values
                    portfolio_dict[title] = stock_list         
                    returns = returns_data.loc[target_day][stock_list].values
                    weights = np.ones(len(stock_list)) / len(stock_list)
                    portfolio_return = np.dot(returns, weights)
                    portfolio_daily_returns.loc[target_day][title] = portfolio_return
                else:
                    stock_list = portfolio_dict[title]
                    returns = returns_data.loc[target_day][stock_list].values
                    weights = np.ones(len(stock_list)) / len(stock_list)
                    portfolio_return = np.dot(returns, weights)
                    portfolio_daily_returns.loc[target_day][title] = portfolio_return

        flag += 1
        transfer = False
    
    portfolio_cumulate_returns = pd.DataFrame(data=None, columns=title_list, index=trade_days[1:])
    for title in title_list:
        portfolio_cumulate_returns[title] = (1 + portfolio_daily_returns[title]).cumprod().values - 1
    
    return portfolio_cumulate_returns

## 两因子检验作图
def plot_2_factor(f1, f2, result):
    fig, axes = plt.subplots(f1, f2)

    colors = plt.get_cmap('Reds', f1)

    for i in range(f1):
        for j in range(f2):
            title = f'Factor1.{i+1}~Factor2.{j+1}'
            axes[i][j].plot(result[title],color=colors(5-i))
            axes[i][j].axhline(0)
            axes[i][j].set_xticks([])
            axes[i][j].set_yticks([])

    fig.text(0.06, 0.5, 'market_cap', va='center', rotation='vertical', fontsize='x-large')
    fig.text(0.5, 0.05, 'return_on_equity_ttm', va='center', ha='center', fontsize='x-large')
```

## 市场因子

按照总市值加权的方式构建投资组合。

如何看待破净股<https://xueqiu.com/3779913185/134853640>

```python
start_year, end_year = 2013, 2023
start_month, end_month = 1, 12

stock_df = pd.DataFrame(data=None, columns=['TRADE_DT', 'TICKER', 'TARGET_WEIGHT'])

first_day = rqdatac.get_trading_dates(start_date='20130101', end_date='20130130')[0]
sub_df = get_target_stock_sub_df(first_day)
stock_df = pd.concat([stock_df, sub_df], axis=0)
```


```python
for year in range(start_year, end_year+1):
    for month in range(start_month, end_month+1):
        _, end = calendar.monthrange(year, month)
        start_day = datetime.date(year, month, 1)
        end_day = datetime.date(year, month, end)
        last_trading_day = rqdatac.get_trading_dates(start_date=start_day, end_date=end_day)[-1]
        sub_df = get_target_stock_sub_df(last_trading_day)
        stock_df = pd.concat([stock_df, sub_df], axis=0)
```


```python
stock_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>TRADE_DT</th>
      <th>TICKER</th>
      <th>TARGET_WEIGHT</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2013-01-04</td>
      <td>000001.XSHE</td>
      <td>0.000140</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2013-01-04</td>
      <td>000002.XSHE</td>
      <td>0.000126</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2013-01-04</td>
      <td>000004.XSHE</td>
      <td>0.000143</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2013-01-04</td>
      <td>000006.XSHE</td>
      <td>0.003145</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2013-01-04</td>
      <td>000007.XSHE</td>
      <td>0.000162</td>
    </tr>
  </tbody>
</table>
</div>




```python
__config__ = {
    "base": {
        "data_bundle_path": r"D:\RQData\bundle",
        "start_date": "20130101",
        "end_date": "20231231",
        "accounts": {
            "stock": 1000000,
        },
    },
    "mod": {
        "sys_analyser": {
            "plot": True
        }
    }
}

def init(context):
    context.target = {d: t.set_index("TICKER")["TARGET_WEIGHT"].to_dict() for d, t in stock_df.groupby("TRADE_DT")}

def handle_bar(context, bar_dict):
    today = datetime.date(context.now.year, context.now.month, context.now.day)
    if today not in context.target:
        return
    order_target_portfolio(context.target[today])
```

```python
from rqalpha_plus import run_func

run_func(init=init, handle_bar=handle_bar, config=__config__)
```

![](/images/basic_factor_research_files/output1.png)

# 规模因子

研究市场上股票市值的分布。

```python
today = datetime.date.today()
all_stock = rqdatac.all_instruments(type='CS',date=today)
query = stock_filter(all_stock)
a_market_stock = all_stock[query]['order_book_id']
```

```python
a_market_cap = rqdatac.get_factor(a_market_stock, 'market_cap', today-datetime.timedelta(1), today-datetime.timedelta(1))
data = a_market_cap['market_cap'].values
```

```python
plt.hist(np.log(data), bins=20, density=True, edgecolor='white')
sns.kdeplot(np.log(data), color='green')
```
    
![png](/images/basic_factor_research_files/basic_factor_research_15_1.png)
    
大部分股票市值位于中小区间，市值分布右偏，大市值公司少。

```python
# 洛伦兹曲线
#-------------------------------
# sort the data in ascending order
var1_sorted = np.arange(len(data))
var2_sorted = np.sort(data)

# calculate the cumulative sum of the sorted data
cumsum_var1 = np.cumsum(var1_sorted)
cumsum_var2 = np.cumsum(var2_sorted)

# normalize the cumulative sum by dividing by the total sum
normalized_cumsum_var1 = cumsum_var1 / np.sum(var1_sorted)
normalized_cumsum_var2 = cumsum_var2 / np.sum(var2_sorted)

# create the perfect equality line
perfect_equality_line = np.linspace(0, 1, len(var1_sorted))

#-------------------------------

# plot the Lorenz curve
plt.plot(normalized_cumsum_var1, normalized_cumsum_var2, label='var1')
plt.plot([0,1], [0,1], label='Perfect equality line', linestyle='--', color='gray')
```
    
![png](/images/basic_factor_research_files/basic_factor_research_17_1.png)
    
以下探究不同指数成分股，市值的分布特点。

```python
def get_components_market_cap_distribution(index, day):
    components = rqdatac.index_components(index, day)
    components_market_cap_df = rqdatac.get_factor(components, 'market_cap', day, day)
    components_market_cap_list = components_market_cap_df['market_cap'].values
    components_market_cap_list = components_market_cap_list[~np.isnan(components_market_cap_list)]
    return components_market_cap_list
```

```python
# 沪深300
g1 = get_components_market_cap_distribution('000300.XSHG', today-datetime.timedelta(1))
# 中证500
g2 = get_components_market_cap_distribution('000905.XSHG', today-datetime.timedelta(1))
# 中证800
g3 = get_components_market_cap_distribution('000906.XSHG', today-datetime.timedelta(1))
# 中证1000
g4 = get_components_market_cap_distribution('000852.XSHG', today-datetime.timedelta(1))
# 中证2000
g5 = get_components_market_cap_distribution('932000.INDX', today-datetime.timedelta(1))
```

```python
plt.rcParams['font.sans-serif'] = ['SimHei']
g = [np.log(g1), np.log(g2), np.log(g3), np.log(g4), np.log(g5)]
plt.boxplot(g, labels=['沪深300','中证500','中证800','中证1000','中证2000'])
```
 
![png](/images/basic_factor_research_files/basic_factor_research_21_1.png)
    

```python
start_date = '20130101'
end_date = '20231231'

f = Factor('market_cap')

all_stock = rqdatac.all_instruments(type='CS',date=today)
query = stock_filter(all_stock)
ids = all_stock[query]['order_book_id']

df = execute_factor(f, ids, start_date, end_date)
df = df.dropna(axis=1)
```


```python
price = rqdatac.get_price(ids, start_date, end_date, frequency='1d', fields='close', expect_df=False)
returns = price.pct_change()
returns.index = pd.DatetimeIndex(returns.index.date)
```


```python
engine = FactorAnalysisEngine()
engine.append(('winzorization-mad', Winzorization(method='mad')))
engine.append(('rank_ic_analysis', ICAnalysis(rank_ic=True, industry_classification='sws')))
engine.append(('QuantileReturnAnalysis', QuantileReturnAnalysis(quantile=10)))
result = engine.analysis(df, returns, ascending=True, periods=22, keep_preprocess_result=True)
```

```python
result['rank_ic_analysis'].show()
```

![](/images/basic_factor_research_files/bokeh_plot1.png)

![](/images/basic_factor_research_files/bokeh_plot2.png)

```python
result['QuantileReturnAnalysis'].show()
```

![](/images/basic_factor_research_files/bokeh_plot4.png)

![](/images/basic_factor_research_files/bokeh_plot5.png)

市值在后20%的小盘股表现非常好。

# 价值因子

所谓价值效应，是相比估值较高的股票，那些估值较低的股票有着更高的预期收益率。

- 账面市值比BM：

    - 账面市值比是衡量公司价值的一项指标。该比率将公司的账面价值与其市场价值进行比较。

    - 账面价值是通过查看公司的历史成本或会计价值来计算的。
    
    - 市场价值取决于其在股票市场的股价和其已发行股票的数量，即其市值。

    - 账面市值比=公司所有者权益总额/公司市值

- 市净率P/B

    - 市净率=每股价格/每股净资产
    
    - 如果P/B等于1，则意味着当前股价和公司账面价值相符，也就是说，这时候的股票价格可以称之为股票公允价值。

    - 如果P/B小于1，则股票可能被低估，也可能意味着该公司的资产回报率(ROA)较差，甚至为负。

    - 如果P/B大于1，则股票可能被高估，也可能意味着该公司的资产回报率(ROA)较高。高股价也可能表明关于公司的利好消息都已反映在股价里了，因此，任何额外的好消息都可能不会导致股价的进一步上涨

- 盈利市值比EP

    - 盈利市值比=归属母公司净利润/总市值

- 市盈率P/E


价值股和成长股：

一、价值股和成长股的定义

1. 价值股
    
    （1）知名度颇高，经过残酷而长的竞争，在各自的领域形成垄断或垄断竞争格局，比如金融、基建、家电等行业中的巨头公司。

    （2）市值规模庞大，不少拥有上千亿或几千亿甚至万亿级别。

    （3）市场占有率高，不少企业在国内某一行业已经占据近50%以上的市场。

    （4）现金流充足，分红优厚。

2. 成长股

    （1）当前处于新兴行业，行业的产生或许是技术驱动，或许是政策引导所致，比如新能源汽车、生物医药、5G、芯片等行业。

    （2）企业收入或利润的增长速度在几年内显著高于传统行业，增速可能达到30%、50%，甚至更高。

    （3）中小市值，行业空间比较大。

    （4）企业处于快速扩张期，会把赚取的利润或大量借来的钱用于投资新的项目或相关企业，导致企业的自由现金流比较紧张。

二、价值股和成长股的特征

- 价值股是指相对于它们的现有业绩收益，股价被低估的一类股票，一般来说具有低市盈率（PE）与市净率（PB）、高股息的特点。

- 成长股是指具有高增长业绩收益，并且市盈率（PE）与市净率（PB）倾向于比其他股票高的一类股票。



## 账面市值比BM

理想情况：账面市值比越高，股票越被低估，股票预期收益更高


```python
start_date = '20130101'
end_date = '20231231'

all_stock = rqdatac.all_instruments(type='CS',date=end_date)
query = stock_filter(all_stock)
ids = all_stock[query]['order_book_id'].values
```


```python
BM_data = rqdatac.get_factor(ids, 'book_to_market_ratio_ttm', start_date, end_date, expect_df=False)
```


```python
engine = FactorAnalysisEngine()
engine.append(('winzorization-mad', Winzorization(method='mad')))
engine.append(('rank_ic_analysis', ICAnalysis(rank_ic=True, industry_classification='sws')))
engine.append(('QuantileReturnAnalysis', QuantileReturnAnalysis(quantile=10)))
result = engine.analysis(BM_data, 'daily', ascending=False, periods=7, keep_preprocess_result=True)
```


```python
result['QuantileReturnAnalysis'].show()
```

![](/images/basic_factor_research_files/bokeh_plot6.png)

![](/images/basic_factor_research_files/bokeh_plot7.png)

![](/images/basic_factor_research_files/bokeh_plot8.png)

## 盈利市值比EP


```python
EP_data = rqdatac.get_factor(ids, 'ep_ratio_ttm', start_date, end_date, expect_df=False)
```


```python
engine = FactorAnalysisEngine()
engine.append(('winzorization-mad', Winzorization(method='mad')))
engine.append(('rank_ic_analysis', ICAnalysis(rank_ic=True, industry_classification='sws')))
engine.append(('QuantileReturnAnalysis', QuantileReturnAnalysis(quantile=10)))
result = engine.analysis(EP_data, 'daily', ascending=False, periods=7, keep_preprocess_result=True)
```


```python
result['QuantileReturnAnalysis'].show()
```

![](/images/basic_factor_research_files/bokeh_plot9.png)

![](/images/basic_factor_research_files/bokeh_plot10.png)

![](/images/basic_factor_research_files/bokeh_plot11.png)

## EP越大，ROA越高

```python
yesterday = datetime.date.today() - datetime.timedelta(1)

EP_data_today = rqdatac.get_factor(ids, 'ep_ratio_ttm', yesterday, yesterday, expect_df=False)
EP_data_today = EP_data_today.dropna().sort_values(ascending=False)
```

```python
roa_mean_list = []

for i in range(10):
    s = int(i/10*len(EP_data_today))
    e = int((i+1)/10*len(EP_data_today))
    group = EP_data_today[s:e].index
    roa = rqdatac.get_factor(group, 'return_on_asset_net_profit_ttm', yesterday, yesterday, expect_df=False)
    roa_mean_list.append(roa.mean())
```


```python
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

plt.plot([f'Group{i}' for i in range(1,11)], roa_mean_list)
plt.title("EP从大到小分组的ROA均值")
```
    
![png](/images/basic_factor_research_files/basic_factor_research_43_1.png)
    


可以看到，EP越大的组，表示估值越低，整体有更高的ROA。

## 两因子检验：EP和市值


```python
all_stock = rqdatac.all_instruments(type='CS',date='20240722')
query = stock_filter(all_stock)
ids = all_stock[query]['order_book_id'].values
ids = list(ids)

f1, f2 = 5, 5
result = Double_Factor_Analysis(ids, 'ep_ratio_ttm', 'market_cap_3', 5, 5, '20130101', '20231231', 22)
```


```python
plot_2_factor(f1, f2, result)
```

![png](/images/basic_factor_research_files/basic_factor_research_47_0.png)
    

从全市场的情况来看：

- 在市值相同时，EP越大，收益越高
- 在EP相同时，市值越小，收益越高

# 动量因子

动量效应：过去涨的股票，接下来很可能会继续涨，“强者恒强。弱者恒弱”。

在A股当中却往往呈现反转效应。

在t月末将t-12到t-1之间的11个月内的总收益作为动量因子的排序变量。


```python
# 自定义动量因子
# 输入某个日期，返回过去11月的总收益率，一个pd.Series，index是股票代码

from dateutil.relativedelta import relativedelta

def get_momentom(order_book_ids, start_date, end_date):
    
    date = start_date
    # 上1个月的月末收盘价
    month_1_before = date - relativedelta(months=1)
    # 上11个月的月初收盘价
    month_11_before = date - relativedelta(months=11)

    # 获取上一个月的最后一个交易日
    ## 先获取上一个月的最后一天 
    month_1_before_year = month_1_before.year
    month_1_before_month = month_1_before.month
    _, month_1_before_end = calendar.monthrange(month_1_before_year, month_1_before_month)
    ## 再获取上一个月的最后一个交易日
    month_1_before_first_day = datetime.datetime(month_1_before_year, month_1_before_month, 1)
    month_1_before_end_day = datetime.datetime(month_1_before_year, month_1_before_month, month_1_before_end)
    month_1_before_last_trading_day = rqdatac.get_trading_dates(month_1_before_first_day, month_1_before_end_day, market='cn')[-1]

    # 获取上一个月的最后一个交易日的收盘价
    month_1_before_close = execute_factor(Factor('close'), ids, month_1_before_last_trading_day, month_1_before_last_trading_day)

    # 获取11个月的第一个交易日
    ## 先获取上11个月的最后一天 
    month_11_before_year = month_11_before.year
    month_11_before_month = month_11_before.month
    _, month_11_before_end = calendar.monthrange(month_11_before_year, month_11_before_month)
    ## 再获取上11个月的第一个交易日
    month_11_before_first_day = datetime.datetime(month_11_before_year, month_11_before_month, 1)
    month_11_before_end_day = datetime.datetime(month_11_before_year, month_11_before_month, month_11_before_end)
    month_11_before_first_trading_day = rqdatac.get_trading_dates(month_11_before_first_day, month_11_before_end_day, market='cn')[0]

    # 获取上一个月的最后一个交易日的收盘价
    month_11_before_close = execute_factor(Factor('close'), ids, month_11_before_first_trading_day, month_11_before_first_trading_day)

    # 计算收益率
    momentom = (month_1_before_close.values - month_11_before_close.values)/month_11_before_close.values
    momentom = momentom.flatten()

    return momentom

```

> 代码有待改进

```python
def momentom(order_book_ids, start_date, end_date):
    trading_days = pd.to_datetime(rqdatac.get_trading_dates(start_date, end_date))
    output = pd.DataFrame(data=None, columns=order_book_ids, index=trading_days)
    for trading_day in trading_days:
        mom = get_momentom(order_book_ids, trading_day, trading_day)
        output.loc[trading_day] = mom
    return output
```


```python
Momentom = UserDefinedLeafFactor('momentom', momentom)
```


```python
start_date = '20130101'
end_date = '20231231'

all_ids = rqdatac.all_instruments(type='CS', date=start_date)

query = stock_filter(all_ids)
ids = list(all_ids[query]['order_book_id'].values)

df = execute_factor(Momentom, ids, start_date, end_date)
```


```python
engine = FactorAnalysisEngine()
engine.append(('winzorization-mad', Winzorization(method='mad')))
engine.append(('rank_ic_analysis', ICAnalysis(rank_ic=True, industry_classification='sws')))
engine.append(('QuantileReturnAnalysis', QuantileReturnAnalysis(quantile=10)))
result = engine.analysis(df, 'daily', ascending=True, periods=22, keep_preprocess_result=True)
```


```python
result['QuantileReturnAnalysis'].show()
```

![](/images/basic_factor_research_files/bokeh_plot12.png)

![](/images/basic_factor_research_files/bokeh_plot13.png)

![](/images/basic_factor_research_files/bokeh_plot14.png)

不满足单调性，此处动量因子反映的动量效应在A股较弱。

# 盈利因子

盈利能力越强，企业前景越好，越容易受到投资者的关注。

- 净资产收益率ROE
    - ROE=归属于股东的利润/净资产

- 总资产收益率ROA
    - ROA=归属股东和债权人的利润/总资产


```python
start_date = '20130101'
end_date = '20231231'

f = Factor('return_on_equity_ttm')

all_ids = rqdatac.all_instruments(type='CS', date=start_date)

query = stock_filter(all_ids)
ids = list(all_ids[query]['order_book_id'].values)

df = execute_factor(f, ids, start_date, end_date)
```


```python
engine = FactorAnalysisEngine()
engine.append(('winzorization-mad', Winzorization(method='mad')))
engine.append(('rank_ic_analysis', ICAnalysis(rank_ic=True, industry_classification='sws')))
engine.append(('QuantileReturnAnalysis', QuantileReturnAnalysis(quantile=10)))
result = engine.analysis(df, 'daily', ascending=False, periods=22, keep_preprocess_result=True)
```


```python
result['QuantileReturnAnalysis'].show()
```

![](/images/basic_factor_research_files/bokeh_plot15.png)

![](/images/basic_factor_research_files/bokeh_plot16.png)

![](/images/basic_factor_research_files/bokeh_plot17.png)

有点奇怪，ROE最小的组反而在大多数时间拿到了最高的收益。。。

跟市值因子一起做两因子检验看看。


```python
result = Double_Factor_Analysis(ids, 'return_on_equity_ttm', 'market_cap_3', 5, 5, '20130101', '20231231', 22)
```


```python
plot_2_factor(5, 5, result)
```


    
![png](/images/basic_factor_research_files/basic_factor_research_65_0.png)
    


排除了市值的影响后，可以看到

- 同一组市值内，都是ROE最高组合实现最高的收益。
- 同一组ROE内，小市值公司收益要高于大市值公司。

# 投资因子

投资效应：当期投资较多的公司相比于投资较少的公司，在未来的预期收益率更低，投资和预期收益之间呈现负相关。

使用总资产同比增长率为排序变量，频率为年度。


```python
start_date = '20130101'
end_date = '20231231'

f = Factor('total_asset_growth_ratio_ttm')

all_ids = rqdatac.all_instruments(type='CS', date=start_date)

query = stock_filter(all_ids)
ids = list(all_ids[query]['order_book_id'].values)

df = execute_factor(f, ids, start_date, end_date)
```


```python
engine = FactorAnalysisEngine()
engine.append(('winzorization-mad', Winzorization(method='mad')))
engine.append(('rank_ic_analysis', ICAnalysis(rank_ic=True, industry_classification='sws')))
engine.append(('QuantileReturnAnalysis', QuantileReturnAnalysis(quantile=10)))
result = engine.analysis(df, 'daily', ascending=True, periods=22, keep_preprocess_result=True)
```


```python
result['QuantileReturnAnalysis'].show()
```

![](/images/basic_factor_research_files/bokeh_plot18.png)

![](/images/basic_factor_research_files/bokeh_plot19.png)

![](/images/basic_factor_research_files/bokeh_plot20.png)

看上去表现尚可。

再做一个双因子检验。


```python
df_result = Double_Factor_Analysis(ids, 'total_asset_growth_ratio_ttm', 'market_cap_3', 5, 5, '20130101', '20231231', 22)
```


```python
plot_2_factor(5, 5, df_result)
```

![png](/images/basic_factor_research_files/basic_factor_research_74_0.png)
    
教材上说，控制了市值后，低投资的月均收益率 低于 高投资的月均收益率。

# 换手率因子

换手率 = 某一段时期内的成交量/发行总股数×100%

教材构建了 异常换手率 ，对于每支股票，在t月末，异常换手率的定义为过去21个交易日的平均换手率和过去252个交易日的平均换手率的比值。


```python
start_date = '20130101'
end_date = '20231231'

f = Factor('VOL20') / Factor('VOL250')

all_ids = rqdatac.all_instruments(type='CS', date=start_date)

query = stock_filter(all_ids)
ids = list(all_ids[query]['order_book_id'].values)

df = execute_factor(f, ids, start_date, end_date)
```


```python
engine = FactorAnalysisEngine()
engine.append(('winzorization-mad', Winzorization(method='mad')))
engine.append(('rank_ic_analysis', ICAnalysis(rank_ic=True, industry_classification='sws')))
engine.append(('QuantileReturnAnalysis', QuantileReturnAnalysis(quantile=10)))
result = engine.analysis(df, 'daily', ascending=True, periods=7, keep_preprocess_result=True)
```


```python
result['QuantileReturnAnalysis'].show()
```

![](/images/basic_factor_research_files/bokeh_plot21.png)

![](/images/basic_factor_research_files/bokeh_plot22.png)

![](/images/basic_factor_research_files/bokeh_plot23.png)

低换手率效应：换手率上升，收益率下降。

从此处的情况来看，这样的效应是存在的。并且换手率因子效果显著，多空差异明显。
