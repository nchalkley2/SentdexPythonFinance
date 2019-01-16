#!/usr/bin/env python3
import datetime as dt
import bs4 as bs
import pickle
import requests
import os.path

import sys
import matplotlib.pyplot as plt
from matplotlib import style
from mpl_finance import candlestick_ohlc
import numpy as np
import matplotlib.dates as mdates
import pandas as pd
import pandas_datareader.data as web

from typing import List, Set, Dict, Tuple, Optional


def save_sp5_tickers() -> List[str]:
    resp = requests.get(
        'http://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    soup = bs.BeautifulSoup(resp.text, 'lxml')
    table = soup.find('table', {'class': 'wikitable sortable'})
    tickers: List[str] = []
    for row in table.findAll('tr')[1:]:
        ticker = row.findAll('td')[0].text
        tickers.append(ticker)

    with open("sp500tickers.pickle", "wb") as f:
        pickle.dump(tickers, f)

    return tickers


def get_sp5_tickers(reload_sp500: bool = False) -> List[str]:
    if reload_sp500 or not os.path.isfile('sp500tickers.pickle'):
        print('loading tickers from wikipedia...')
        return save_sp5_tickers()
    else:
        print('loading tickers from file...')
        return pickle.load(open('sp500tickers.pickle', 'rb'))


def get_data_from_yahoo(reload_sp500: bool = False,
                        tickers: List[str] = None) -> Dict[str, pd.DataFrame]:
    if tickers is None:
        tickers = get_sp5_tickers(reload_sp500)

    if not os.path.exists('stock_dfs'):
        os.makedirs('stock_dfs')

    stocks = {}
    for symbol in tickers:
        symbol = symbol.replace('.', '-')
        path = 'stock_dfs/{}.csv'.format(symbol)
        if not os.path.exists(path) or reload_sp500:
            print('downloading symbol: {}'.format(symbol))
            try:
                df: pd.DataFrame = web.DataReader(symbol, 'yahoo')
                df.to_csv(path)
                stocks[symbol] = df
            except Exception as e:
                print('error: ', e)
                print('failed to download symbol: {}'.format(symbol))
        else:
            print('loading {} from file...'.format(symbol))
            stocks[symbol] = pd.read_csv(path)

    return stocks


def compile_data(reload_sp500: bool = False,
                 ticker_dfs: Dict[str, pd.DataFrame] = None) -> pd.DataFrame:
    if ticker_dfs is None:
        ticker_dfs = get_data_from_yahoo(reload_sp500)

    main_df = pd.DataFrame()

    for ticker, df in ticker_dfs.items():
        print(ticker)
        df.set_index('Date', inplace=True)
        df.rename(columns={'Adj Close': ticker}, inplace=True)
        df.drop(['Open', 'High', 'Low', 'Close', 'Volume'], 1, inplace=True)

        main_df = df if main_df.empty else main_df.join(df, how='outer')

    main_df.to_csv('sp500_joined_closes.csv')
    return main_df


def get_sp500_df(reload_sp500: bool = False) -> pd.DataFrame:
    if reload_sp500 or not os.path.isfile('sp500_joined_closes.csv'):
        print('compiling sp500 dataframe...')
        return compile_data(reload_sp500)
    else:
        print('loading sp500 dataframe from file...')
        return pd.read_csv('sp500_joined_closes.csv')


def visualize_data(sp500_df: pd.DataFrame) -> None:
    sp500_df_corr = sp500_df.corr()
    data: np.ndarray = sp500_df_corr.values
    fig: plt.Figure = plt.figure()
    ax: plt.Axes = fig.add_subplot(1, 1, 1)
    
    heatmap = ax.pcolor(data, cmap=plt.viridis())
    fig.colorbar(heatmap)
    ax.set_xticks(np.arange(data.shape[1]) + 0.5, minor=False)
    ax.set_yticks(np.arange(data.shape[0]) + 0.5, minor=False)
    ax.invert_yaxis()
    ax.xaxis.tick_top()

    column_labels = sp500_df_corr.columns
    row_labels = sp500_df_corr.index

    ax.set_xticklabels(column_labels)
    ax.set_yticklabels(row_labels)
    plt.xticks(rotation=90)
    heatmap.set_clim(-1,1)
    plt.tight_layout()
    plt.show()

    #print(sp500_df_corr)

def process_data_for_labels(sp500_df: pd.DataFrame, ticker):
    hm_days = 7
    tickers = sp500_df.columns.values.tolist()
    sp500_df.fillna(0, inplace=False)    


df = get_sp500_df()
visualize_data(df)