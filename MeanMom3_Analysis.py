from termcolor import colored
import tdaClientInterpreter as tda
from datetime import datetime, timedelta
import traceback
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
import statsmodels.api as sm
from statsmodels.tsa.ar_model import AutoReg


def PlotRecent():
    recent_data = pd.read_csv("Resources/Output/recent_data_15.csv", index_col=0)
    qqq_data = pd.read_csv("Resources/Data/QQQ_daily_data.csv", index_col=0)

    qqq_data['pChange'] = qqq_data['close'] / qqq_data['close'].shift(1)
    qqq_data['total_pChange'] = qqq_data['pChange'].rolling(len(qqq_data.index), min_periods=1).apply(np.prod)
    qqq_data['Balance'] = qqq_data['total_pChange'] * 100000
    qqq_data['log_balance'] = np.log(qqq_data['Balance'])
    qqq_data.dropna(inplace=True)

    print(recent_data)
    print(qqq_data)

    fig, ax1 = plt.subplots()

    ax1.plot(recent_data.index, recent_data.Balance, c='green', label='Algorithm')
    ax1.plot(qqq_data.index, qqq_data.Balance, c='blue', label='Market')

    plt.legend()
    plt.show()


def SharpeRatio():
    recent_data = pd.read_csv("Resources/Output/Track_Data_recent.csv", index_col=0)
    qqq_data = pd.read_csv("Resources/Data/QQQ_daily_data.csv", index_col=0)
    qqq_data['pChange'] = (qqq_data['close'] - qqq_data['close'].shift(1)) / qqq_data['close'].shift(1)

    avg_pChange = recent_data['pChange'].mean()
    std_dev_algo = recent_data['pChange'].std()
    qqq_avg_pChange = qqq_data['pChange'].mean()
    std_dev_qqq = qqq_data['pChange'].std()

    sharpe_algo = (avg_pChange / std_dev_algo) * math.sqrt(252)
    sharpe_qqq = (qqq_avg_pChange / std_dev_qqq) * math.sqrt(252)
    print(f"{sharpe_algo = }")
    print(f"{sharpe_qqq = }")


def SortinoRatio():
    recent_data = pd.read_csv("Resources/Output/Track_Data_recent.csv", index_col=0)
    qqq_data = pd.read_csv("Resources/Data/QQQ_daily_data.csv", index_col=0)
    qqq_data['pChange'] = (qqq_data['close'] - qqq_data['close'].shift(1)) / qqq_data['close'].shift(1)

    avg_pChange = recent_data['pChange'].mean()
    recent_data['downside_pChange'] = recent_data[recent_data['pChange'] < 0]['pChange']
    recent_data['downside_pChange'] = recent_data['downside_pChange'].fillna(0)
    std_dev_algo = recent_data['downside_pChange'].std()

    qqq_avg_pChange = qqq_data['pChange'].mean()
    qqq_data['downside_pChange'] = qqq_data[qqq_data['pChange'] < 0]['pChange']
    qqq_data['downside_pChange'] = qqq_data['downside_pChange'].fillna(0)
    std_dev_qqq = qqq_data['downside_pChange'].std()

    sortino_algo = (avg_pChange / std_dev_algo) * math.sqrt(252)
    sortino_qqq = (qqq_avg_pChange / std_dev_qqq) * math.sqrt(252)
    print(f"{sortino_algo = }")
    print(f"{sortino_qqq = }")


def SuccessRate():
    recent_data = pd.read_csv("Resources/Output/Track_Data_recent.csv", index_col=0)
    success_data = recent_data[recent_data['pChange'] > 0]
    print(len(success_data.index) / len(recent_data.index))

    qqq_data = pd.read_csv("Resources/Data/QQQ_daily_data.csv", index_col=0)
    qqq_data['pChange'] = (qqq_data['close'] - qqq_data['close'].shift(1)) / qqq_data['close'].shift(1)
    qqq_success_data = qqq_data[qqq_data['pChange'] > 0]

    print(len(qqq_success_data.index) / len(qqq_data.index))


def BetaRatio():
    recent_data = pd.read_csv("Resources/Output/Track_Data_recent.csv")
    recent_data.drop(0, inplace=True)
    start_date = recent_data.iloc[0]['Unnamed: 0']
    recent_data.drop(1, inplace=True)
    recent_data = recent_data.reset_index()

    qqq_data = pd.read_csv("Resources/Data/QQQ_daily_data.csv", index_col=0)
    qqq_data['pChange'] = (qqq_data['close'] - qqq_data['close'].shift(1)) / qqq_data['close'].shift(1)
    qqq_data = qqq_data[start_date:].reset_index()

    print(recent_data)
    print(qqq_data)
    x = recent_data[['pChange']].astype(float)
    x = sm.add_constant(x)

    y = qqq_data['pChange'].astype(float)

    regr = sm.OLS(y, x)
    results = regr.fit()
    print(results.summary())


def CorrelationAnalysis():
    recent_data = pd.read_csv("Resources/Output/Track_Data_recent.csv")
    recent_data.drop(0, inplace=True)

    recent_data['success'] = np.where(recent_data['pChange'] > 0, 1, 0)

    print(recent_data)
    recent_data.replace([np.inf, -np.inf], np.nan, inplace=True)
    recent_data.dropna(inplace=True)

    x = recent_data[['score', 'rsquared', 'qqq_pChange']].astype(float)
    # x = recent_data[['Expected_Returns']].astype(float)
    x = sm.add_constant(x)
    print(x)

    y = recent_data[['success']].astype(float)
    print(y)

    log_regr = sm.Logit(y, x)
    regr = sm.OLS(y, x)
    log_results = log_regr.fit()
    results = regr.fit()

    print(log_results.summary())
    print(results.summary())

    fig = sm.graphics.plot_fit(results, 1)
    plt.show()


def AutoRegressionAnalysis():
    recent_data = pd.read_csv("Resources/Output/Track_Data_recent.csv")
    recent_data.drop(0, inplace=True)

    # recent_data['score'] = np.abs(recent_data['score'])
    print(recent_data)
    recent_data.replace([np.inf, -np.inf], np.nan, inplace=True)

    recent_data.dropna(inplace=True)

    res = AutoReg(recent_data['pChange'], lags=3).fit()

    print(res.summary())


def MaxDrawDown():
    recent_data = pd.read_csv("Resources/Output/Track_Data_recent.csv", index_col=0)
    rolling_max = recent_data['Balance'].rolling(len(recent_data.index), min_periods=1).max()
    daily_drawdown = recent_data['Balance'] / rolling_max - 1
    print("Max Drawdown Algorithm:", daily_drawdown.min())

    qqq_data = pd.read_csv("Resources/Data/QQQ_daily_data.csv", index_col=0)
    qqq_data['pChange'] = qqq_data['close'] / qqq_data['close'].shift(1)
    qqq_data['total_pChange'] = qqq_data['pChange'].rolling(len(qqq_data.index), min_periods=1).apply(np.prod)
    qqq_data['Balance'] = qqq_data['total_pChange'] * 100000

    rolling_max_qqq = qqq_data['Balance'].rolling(len(qqq_data.index), min_periods=1).max()
    daily_drawdown_qqq = qqq_data['Balance'] / rolling_max_qqq - 1
    print("Max Drawdown QQQ:", daily_drawdown_qqq.min())


if __name__ == "__main__":
    # CorrelationAnalysis()
    SortinoRatio()
    SharpeRatio()
    SuccessRate()
    BetaRatio()
    MaxDrawDown()
