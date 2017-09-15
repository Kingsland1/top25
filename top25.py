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


def main(hold_days=1, stock_num=25, pick_window=1):
    """
    打板策略
    """
    data = pd.read_csv(filepath_or_buffer="ZZ500.csv", index_col=0, header=1)
    stock_columns = list(pd.read_csv(filepath_or_buffer="ZZ500.csv", index_col=0, nrows=1).columns)[::7]
    data_open = data.iloc[:, ::7].astype(float)
    data_close = data.iloc[:, 3::7].astype(float)
    data_open.columns = stock_columns
    data_close.columns = stock_columns
    top_stocks = get_stocks(data_close, stock_num, pick_window)
    return_holding = get_return_holding(data_open, top_stocks, pick_window, hold_days, stock_num)
    # 把回报matrix row_wise 求平均，再计算累计积
    avg = return_holding.mean(axis=1)
    cumsum = (avg + 1).cumprod()
    cumsum.plot()
    plt.figure()
    return_holding.mean(axis=1).plot.hist(bins=100)
    plt.show()


def get_stocks(data_close, stock_num, pick_window):
    """
    得到每天的选股，stock_num个股票，第1天和第t天的闭盘收益比，得到第(t+1)天的选股
    """
    performance = data_close.pct_change(periods=pick_window)
    date_index = data_close.index
    top_stocks = pd.DataFrame(index=date_index, columns=range(stock_num))
    for i in range(pick_window, len(date_index) - 1):
        top_stocks.iloc[i + 1] = pd.Series(performance.iloc[i]).sort_values(ascending=False).index[:stock_num]
    return top_stocks


def get_return_holding(data_open, top_stocks, pick_window, hold_days, stock_num):
    """
    找到top25股票名字 所对应的隔日回报，比如今天开盘买进，明天开盘卖出
    """
    return_open = data_open.pct_change(periods=hold_days)
    date_index = data_open.index
    return_holding = pd.DataFrame(index=date_index, columns=range(stock_num))
    for i in range(1 + pick_window, len(date_index) - hold_days):
        date1 = date_index[i]
        date2 = date_index[i + hold_days]
        return_holding.loc[date2] = pd.Series(return_open.loc[date2, top_stocks.loc[date1]].values)

    return return_holding

if __name__ == "__main__":
    if len(sys.argv) == 1:
        main()
    elif len(sys.argv) == 4:
        hold_days = int(sys.argv[1])
        stock_num = int(sys.argv[2])
        pick_window = int(sys.argv[3])
        main(hold_days, stock_num, pick_window)
    else:
        print("args error", file=sys.stderr)
