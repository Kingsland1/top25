#算出中证500股票的每日收益   (close/open)-1
#每天，对收益进行从高到低排序，找到Top25的股票，作为第二天的持仓，等权---就会产生一个每日仓位的 Dataframe
#算出所有股票，每日的close/open-1 作为收益指标
# 用T天算出的Top25的股票，作为T+1的持仓，其T+1的收益 为策略当天的收益，等权（暂时忽略了股票不能进行T0交易的限制）
# 每天收益的+1,再算cumprod为总收益，画出走势图，画出每日收益分布图


#to do list
#



import pandas as pd
# import rolling_beta
import matplotlib.pyplot as plt

# writer = pd.ExcelWriter('test.xlsx')


data = pd.DataFrame.from_csv("ZZ500.csv")
columns=list(data.columns)[::7]
data_open=data.iloc[1:,::7].astype(float)
data_high=data.iloc[1:,1::7].astype(float)
data_low=data.iloc[1:,2::7].astype(float)
data_close=data.iloc[1:,3::7].astype(float)
data_volume=data.iloc[1:,4::7].astype(float)
data_open.columns = columns
data_high.columns=columns
data_low.columns=columns
data_close.columns=columns
data_volume.columns=columns
return_close=data_close.pct_change()
return_open=data_open.pct_change()
return_intraday=(data_close-data_open)/data_open

# 找到top25的股票的名字
top25=pd.DataFrame(index=data_open.index,columns=range(25))
for date in return_close.index:
    top25.loc[date]=pd.Series(return_close.loc[date]).sort_values(ascending=False).index[:25]

# 找到top25股票名字 所对应的隔日回报，比如今天开盘买进，明天开盘卖出
top25_return_holding=pd.DataFrame(index=data_open.index,columns=columns )
for i,date in enumerate(return_open.index[:-1]):
     top25_return_holding.loc[date]=return_open.loc[return_open.index[i + 1], top25.iloc[i - 1, :]]
# 把回报matrix row_wise 求平均，再累加起来
top25_return_holding_avg=pd.DataFrame(top25_return_holding.mean(axis=1))
top25_return_holding_avg_cumsum=pd.DataFrame(top25_return_holding_avg+1).cumprod()

top25_return_holding_avg_cumsum.plot()
plt.figure()
top25_return_holding.mean(axis=1).plot.hist(bins=100)
plt.show()
