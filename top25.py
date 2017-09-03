#!/usr/bin/env python
# -*- coding: utf-8 -*-
########################################################################
# 
# Copyright (c) 2017 Kingsland1.com, Inc. All Rights Reserved
# 
########################################################################
 
"""
File: top25.py
Author: Kingsland1(fantine16@163.com)
Date: 2017/09/03 14:06:46
"""
import sys
import pandas as pd
import matplotlib.pyplot as plt


def main(stock_num=25):
    """
    打板策略
    """
    data = pd.read_csv(filepath_or_buffer="ZZ500.csv", index_col=0, header=1)
    stock_columns = list(pd.read_csv(filepath_or_buffer="ZZ500.csv", index_col=0, nrows=1).columns)[::7]
    date_index = data.index
    data_open = data.iloc[:, ::7].astype(float)
    data_close = data.iloc[:, 3::7].astype(float)
    data_open.columns = stock_columns
    data_close.columns = stock_columns
    return_open = data_open.pct_change()
    return_close = data_close.pct_change()
    # 得到每天的选股，25个股票，第一天和第二天的闭盘收益比，得到第三天的选股
    top_stocks = pd.DataFrame(index=date_index, columns=range(stock_num))
    for i, date in enumerate(date_index[:-2]):
        top_stocks.iloc[i+2] = pd.Series(return_close.iloc[i+1]).sort_values(ascending=False).index[:stock_num]
    # 找到top25股票名字 所对应的隔日回报，比如今天开盘买进，明天开盘卖出
    return_holding = pd.DataFrame(index=date_index, columns=range(stock_num))
    for i, date in enumerate(date_index[2:-1]):
        return_holding.loc[date] = pd.Series(return_open.loc[date_index[2:][i + 1], top_stocks.loc[date]].values)
    # 把回报matrix row_wise 求平均，再计算累计积
    avg = return_holding.mean(axis=1)
    cumsum = (avg + 1).cumprod()
    cumsum.plot()
    plt.figure()
    return_holding.mean(axis=1).plot.hist(bins=100)
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) == 1:
        main(25)
    elif len(sys.argv) == 2:
        main(int(sys.argv[1]))
    else:
        print("args error", file=sys.stderr)

