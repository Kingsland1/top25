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
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import copy



def main(args):
    """
    打板策略
    """
    data = pd.read_csv(filepath_or_buffer="ZZ500.csv", index_col=0, header=1)
    stock_columns = list(pd.read_csv(filepath_or_buffer="ZZ500.csv", index_col=0, nrows=1).columns)[::7]
    data_open = data.iloc[:, ::7].astype(float)
    data_close = data.iloc[:, 3::7].astype(float)
    data_open.columns = stock_columns
    data_close.columns = stock_columns
    df1 = watch_stock(data_close, data_open, args)
    df2 = watch_hold_days(data_close, data_open, args)
    df3 = watch_pick_window(data_close, data_open, args)
    df1.plot(title=args).get_figure().savefig('df1.png')
    df2.plot(title=args).get_figure().savefig('df2.png')
    df3.plot(title=args).get_figure().savefig('df3.png')
    plt.plot()


def watch_stock(data_close, data_open, args, stock_range=range(15, 50, 3)):
    """
    固定hold_days和pick_window，观察stock_num
    """
    date_index = data_open.index
    stock_range1 = ['stock_num=' + str(i) for i in stock_range]
    df = pd.DataFrame(index=date_index, columns=stock_range1)
    temp_args = copy.copy(args)
    for i in stock_range:
        temp_args.stock_num = i
        cumsum = strategy(data_close, data_open, temp_args)
        df['stock_num=' + str(i)] = cumsum
    return df


def watch_hold_days(data_close, data_open, args, hold_range=range(1, 5)):
    """
    固定stock_num和pick_window，观察hold_days
    """
    date_index = data_open.index
    hold_range1 = ['hold_days=' + str(i) for i in hold_range]
    df = pd.DataFrame(index=date_index, columns=hold_range1)
    temp_args = copy.copy(args)
    for i in hold_range:
        temp_args.hold_days = i
        cumsum = strategy(data_close, data_open, temp_args)
        df['hold_days=' + str(i)] = cumsum
    return df


def watch_pick_window(data_close, data_open, args, pick_range=range(1, 4)):
    """
    固定hold_days和stock_num，观察pick_window
    """
    date_index = data_open.index
    pick_range1 = ['pick_window=' + str(i) for i in pick_range]
    df = pd.DataFrame(index=date_index, columns=pick_range1)
    temp_args = copy.copy(args)
    for i in pick_range:
        temp_args.pick_window = i
        cumsum = strategy(data_close, data_open, temp_args)
        df['pick_window=' + str(i)] = cumsum
    return df


def strategy(data_close, data_open, args):
    """
    根据一组参数args，得到的收益
    """
    top_stocks = get_stocks(data_close, args)
    return_holding = get_return_holding(data_open, top_stocks, args)
    avg = return_holding.mean(axis=1)
    cumsum = (avg + 1).cumprod()
    return cumsum


def get_stocks(data_close, args):
    """
    得到每天的选股，stock_num个股票，第1天和第t天的闭盘收益比，得到第(t+1)天的选股
    """
    stock_num = args.stock_num
    pick_window = args.pick_window
    performance = data_close.pct_change(periods=pick_window)
    date_index = data_close.index
    top_stocks = pd.DataFrame(index=date_index, columns=range(stock_num))
    for i in range(pick_window, len(date_index) - 1):
        top_stocks.iloc[i + 1] = pd.Series(performance.iloc[i]).sort_values(ascending=False).index[:stock_num]
    return top_stocks


def get_return_holding(data_open, top_stocks, args):
    """
    找到top25股票名字 所对应的隔日回报，比如今天开盘买进，明天开盘卖出
    """
    stock_num = args.stock_num
    pick_window = args.pick_window
    hold_days = args.hold_days
    return_open = data_open.pct_change(periods=hold_days)
    date_index = data_open.index
    return_holding = pd.DataFrame(index=date_index, columns=range(stock_num))
    for i in range(1 + pick_window, len(date_index) - hold_days):
        date1 = date_index[i]
        date2 = date_index[i + hold_days]
        return_holding.loc[date2] = pd.Series(return_open.loc[date2, top_stocks.loc[date1]].values)
    return return_holding




if __name__ == "__main__":
    """
    程序入口
    """
    parser = argparse.ArgumentParser(description='打板策略')
    parser.add_argument('-hd', '--hold_days', type=int, default=3)
    parser.add_argument('-sn', '--stock_num', type=int, default=15)
    parser.add_argument('-pw', '--pick_window', type=int, default=1)
    args = parser.parse_args()
    main(args)


