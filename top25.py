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
import numpy as np
import matplotlib.pyplot as plt
import itertools
import copy
import timeit


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

    a = new_strategy(data_close, data_open, args)
    df = a.holding.nav.iloc[:, 2]
    df.plot(title=args).get_figure().savefig('df.png')
    df1 = new_watch_stock(data_close, data_open, args)
    df2 = new_watch_hold_days(data_close, data_open, args)
    df3 = new_watch_pick_window(data_close, data_open, args)
    # df1 = watch_stock(data_close, data_open, args)
    # df2 = watch_hold_days(data_close, data_open, args)
    # df3 = watch_pick_window(data_close, data_open, args)

    df1.plot(title=args).get_figure().savefig('df1.png')
    df2.plot(title=args).get_figure().savefig('df2.png')
    df3.plot(title=args).get_figure().savefig('df3.png')
    plt.plot()


def new_watch_stock(data_close, data_open, args, stock_range=range(10, 31, 3)):
    """
    固定hold_days和pick_window，观察stock_num
    """

    date_index = data_open.index
    stock_range1 = ['stock_num=' + str(i) for i in stock_range]
    df = pd.DataFrame(index=date_index, columns=stock_range1)
    temp_args = copy.copy(args)
    for i in stock_range:
        temp_args.stock_num = i
        portfolio = new_strategy(data_close, data_open, temp_args)
        df['stock_num=' + str(i)] = portfolio.holding.nav.iloc[:, 2]
    return df


def watch_stock(data_close, data_open, args, stock_range=range(1, 16, 3)):
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


def new_watch_hold_days(data_close, data_open, args, hold_range=range(5, 21, 5)):
    """
    固定stock_num和pick_window，观察hold_days
    """
    date_index = data_open.index
    hold_range1 = ['hold_days=' + str(i) for i in hold_range]
    df = pd.DataFrame(index=date_index, columns=hold_range1)
    temp_args = copy.copy(args)
    for i in hold_range:
        temp_args.hold_days = i
        portfolio = new_strategy(data_close, data_open, temp_args)
        df['stock_num=' + str(i)] = portfolio.holding.nav.iloc[:, 2]
    return df


def watch_hold_days(data_close, data_open, args, hold_range=range(5, 21, 5)):
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


def new_watch_pick_window(data_close, data_open, args, pick_range=range(1, 4)):
    """
    固定hold_days和stock_num，观察pick_window
    """
    date_index = data_open.index
    pick_range1 = ['pick_window=' + str(i) for i in pick_range]
    df = pd.DataFrame(index=date_index, columns=pick_range1)
    temp_args = copy.copy(args)
    for i in pick_range:
        temp_args.pick_window = i
        portfolio = new_strategy(data_close, data_open, temp_args)
        df['stock_num=' + str(i)] = portfolio.holding.nav.iloc[:, 2]
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


class Portfolio:
    def __init__(self, data_close, args):
        self.trading = Trading(data_close, args)
        self.holding = Holding(data_close)
        self.current = Current()


class Trading:
    def __init__(self, data_close, args):
        self.buy_name = pd.DataFrame(index=data_close.index)
        self.sell_name = pd.DataFrame(index=data_close.index)
        self.buy_price = pd.DataFrame(index=data_close.index)
        self.sell_price = pd.DataFrame(index=data_close.index)
        self.buy_amount = pd.DataFrame(index=data_close.index, columns=range(args.stock_num))
        self.sell_amount = pd.DataFrame(index=data_close.index, columns=range(args.stock_num))


class Holding:
    def __init__(self, data_close):
        self.cash = pd.DataFrame(index=data_close.index, columns=['open_cash', 'cash_in', 'cash_out', 'close_cash'])
        self.current_holding = pd.DataFrame(index=data_close.index)
        self.amount = pd.DataFrame(index=data_close.index)
        self.price_table = pd.DataFrame()
        self.cost_table = pd.DataFrame()
        self.nav = pd.DataFrame(index=data_close.index, columns=['close_cash', 'stock_value', 'nav'])


class Current:
    def __init__(self):
        """
        holding is [[name],[],[amount]]
        """
        self.date = str()
        self.cash = float()
        self.holding = {}
        self.market_value = float()


def get_price_table(Dataframe, price_source):  # 给一个股票名字的df,返回这些股票的价格
    price_table = pd.DataFrame(index=Dataframe.index,
                               columns=range(Dataframe.shape[1]))
    for i, date in enumerate(Dataframe.index):
        for j, stock in enumerate(Dataframe.loc[date]):
            if pd.notnull(stock) == 1:
                price_table.iloc[i, j] = price_source.loc[date, stock]
    return price_table


def get_buy_amount(budget, trading_list):  # 通过交易清单，计算出买卖的股票的数量
    """
    trading list is a [[name],[price],[amount]]
    """
    budget_per_stock = budget / len(trading_list[1])
    trading_list[2] = [0] * len(trading_list[0])

    for i, stock in enumerate(trading_list[0]):
        if pd.notnull(stock) == 1:
            trading_list[2][i] = budget_per_stock / trading_list[1][i]
    return trading_list[2]


def get_current_holding(Portfolio, args):
    current_holding_table = pd.DataFrame(index=Portfolio.trading.buy_name.index,
                                         columns=range(args.hold_days * Portfolio.trading.buy_name.shape[1]))
    for i, date in enumerate(Portfolio.trading.buy_name.index):

        stock_block = Portfolio.trading.buy_name.iloc[max(i + 1 - args.hold_days, 0):i + 1, :].values.tolist()
        current_holding_table1 = list(itertools.chain.from_iterable(stock_block))
        start_time = timeit.default_timer()
        for j, stock in enumerate(current_holding_table1):
            current_holding_table.iloc[i, j] = stock
        print(timeit.default_timer() - start_time)
    return current_holding_table


def get_holding_amount(Portfolio, args):
    amount_table = pd.DataFrame(index=Portfolio.trading.buy_name.index,
                                columns=range(args.hold_days * Portfolio.trading.buy_name.shape[1]))
    for i, date in enumerate(Portfolio.trading.buy_amount.index):
        amount_block = np.array(Portfolio.trading.buy_amount.iloc[max(i + 1 - args.hold_days, 0):i + 1, :]).tolist()
        amount_table1 = list(itertools.chain.from_iterable(amount_block))
        for j, stock in enumerate(amount_table1):
            amount_table.iloc[i, j] = stock
    return amount_table


def get_holding_nav(Portfolio, args):
    nav_table = pd.DataFrame(index=Portfolio.holding.cash.index, columns=['close_cash', 'stock_value', 'nav'])
    for i, date in enumerate(Portfolio.trading.buy_amount.index):
        nav_table.iloc[i, 0] = Portfolio.holding.cash.iloc[i, 3]
        nav_table.iloc[i, 1] = np.nansum(Portfolio.holding.amount.iloc[i] * Portfolio.holding.price_table.iloc[i])
    nav_table.iloc[:, 2] = nav_table.iloc[:, 0] + nav_table.iloc[:, 1]

    return nav_table


def trade_portfolio(Portfolio, args):
    Portfolio.current.cash = float(args.asset)

    for i, date in enumerate(Portfolio.trading.buy_name.index):

        """    
        trading list is a [[name],[price],[amount]]     
        """
        """
        卖股票
        只写了卖掉全部股票的程序，还可以扩展成没有完全卖空的情况
        """
        sell_list = [[], [], []]
        sell_list[0] = np.array(Portfolio.trading.sell_name.iloc[i, :].values)
        sell_list[1] = np.array(Portfolio.trading.sell_price.iloc[i, :].values)
        sell_list[2] = np.array(Portfolio.trading.buy_amount.iloc[i - args.hold_days, :])
        cash_in = np.nansum(sell_list[1] * sell_list[2])
        if np.isnan(cash_in) == 1:
            pass
        else:

            """cash open"""
            Portfolio.holding.cash.iloc[i, 0] = Portfolio.current.cash
            Portfolio.holding.cash.iloc[i, 1] = cash_in
            Portfolio.current.cash = Portfolio.current.cash + cash_in

        """
        买股票
        """
        buy_list = [[], [], []]
        buy_list[0] = Portfolio.trading.buy_name.iloc[i, :].values
        buy_list[1] = Portfolio.trading.buy_price.iloc[i, :].values
        buy_list[2] = get_buy_amount(budget=min(Portfolio.current.cash, args.asset / args.hold_days),
                                     trading_list=buy_list)
        cash_out = np.nansum(buy_list[1] * buy_list[2])
        if np.isnan(cash_out) == 1:
            pass
        else:
            Portfolio.holding.cash.iloc[i, 2] = cash_out
            Portfolio.current.cash = Portfolio.current.cash - cash_out
            """cash close"""
            Portfolio.holding.cash.iloc[i, 3] = Portfolio.current.cash

        """
        记录交易结果
        """
        Portfolio.trading.buy_amount.iloc[i, :] = buy_list[2]
        Portfolio.trading.sell_amount.iloc[i, :] = sell_list[2]

    return Portfolio


def new_strategy(data_close, data_open, args):
    a = Portfolio(data_close, args)
    a.trading.buy_name = get_stocks(data_close, args)
    a.trading.sell_name = a.trading.buy_name.shift(args.hold_days)
    a.trading.buy_price = get_price_table(a.trading.buy_name, data_open)
    a.trading.sell_price = get_price_table(a.trading.sell_name, data_open)
    a = trade_portfolio(a, args)
    a.holding.current_holding = get_current_holding(a, args)
    a.holding.amount = get_holding_amount(a, args)
    # a.holding.cost_table = get_price_table(a.holding.current_holding, data_open)
    a.holding.price_table = get_price_table(a.holding.current_holding, data_close)
    a.holding.nav = get_holding_nav(a, args)
    return a


if __name__ == "__main__":
    """
    程序入口
    """
    parser = argparse.ArgumentParser(description='打板策略')
    parser.add_argument('-hd', '--hold_days', type=int, default=10)
    parser.add_argument('-sn', '--stock_num', type=int, default=15)
    parser.add_argument('-pw', '--pick_window', type=int, default=1)
    parser.add_argument('-a', '--asset', type=int, default=10000000)
    args = parser.parse_args()
    main(args)
