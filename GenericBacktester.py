from termcolor import colored
import tdaClientInterpreter as tda
from datetime import datetime, timedelta
import traceback
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse
from AlgorithmInterface import Algorithm, Wallet


class Backtest:
    def __init__(self, start, end):
        self.tickers = ["QQQ", "AAPL", "ADBE", "ADI", "ADSK", "AEP", "ALGN", "AMAT", "AMD",
                        "AMGN", "AMZN", "ANSS", "ASML", "ATVI", "AVGO", "BIDU", "BKNG",
                        "CDNS", "CDW", "CERN", "CHKP", "CMCSA", "COST",
                        "CRWD", "CSCO", "CSX", "CTAS", "CTSH", "DLTR", "DXCM",
                        "EA", "EBAY", "EXC", "FAST", "FB", "FISV", "FOX", "GILD",
                        "GOOG", "HON", "IDXX", "ILMN", "INCY", "INTC", "INTU", "ISRG",
                        "KLAC", "LRCX", "LULU", "MAR", "MCHP", "MDLZ",
                        "MELI", "MNST", "MRVL", "MSFT", "MU", "NFLX",
                        "NTES", "NVDA", "NXPI", "ORLY", "PAYX", "PCAR",
                        "PEP", "PTON", "QCOM", "REGN", "ROST", "SBUX", "SGEN",
                        "SIRI", "SNPS", "SPLK", "SWKS", "TCOM", "TMUS", "TSLA",
                        "TXN", "VRSK", "VRSN", "WBA", "WDAY", "XEL", "XLNX", "ZM"]
                        #"JD", "KHC", "PYPL", "MTCH", "TEAM", "OKTA", "DOCU", "PDD", "MRNA"]

        self.QQQ = pd.read_csv("Resources/Data/QQQ_daily_data.csv", index_col=0,
                               parse_dates=['datetime'])

        self.startDatetime = start
        self.endDatetime = end

        self.tickerData = {}
        for ticker in self.tickers:
            try:
                self.tickerData[ticker] = pd.read_csv(f"Resources/Data/{ticker}_daily_data.csv", index_col=0,
                                                  parse_dates=['datetime'])
            except FileNotFoundError:
                self.tickers.remove(ticker)

        # Init Variables:
        self.WarmUp = 10

    def PreProcess(self):
        self.QQQ['pChange'] = self.QQQ['close'] / self.QQQ['close'].shift(1)

        for ticker, data in self.tickerData.items():
            data['pChange_ratio'] = data['close'] / data['close'].shift(1)
            data['pChange'] = 1 - data['pChange_ratio']
            data['velocity'] = data['pChange'] * np.log(data['volume'])

            data.dropna(inplace=True)

    def RunBacktest(self, algo: Algorithm):
        delta_day = timedelta(days=1)
        delta_hour = timedelta(hours=1)

        idxCount = 0
        for idx, row in tqdm(self.QQQ.iterrows(), total=len(self.QQQ.index)):
            idx = idx + delta_hour

            if idxCount <= self.WarmUp:
                idxCount += 1
                continue

            for ticker, data in self.tickerData.items():
                for hr_adjust in range(0, 4):
                    if idx - (hr_adjust * delta_hour) in data.index:
                        algo.tickerData[ticker] = data.loc[:idx - (hr_adjust * delta_hour)]

                        currentSlice = data.loc[idx - (hr_adjust * delta_hour)]
                        algo.currentSlice[ticker] = currentSlice
                        break
                else:
                    algo.currentSlice[ticker] = None

            algo.now = idx
            # algo.wallet.now = idx
            algo.OnData()
            # algo.wallet.UpdateData(algo.currentSlice)


def ExportDataToCSV(tickers, startdate, enddate):
    for s in tickers:
        try:
            temppd = tda.get_price_history_day(s, startdate, enddate)
            temppd['datetime'] = temppd['datetime'] - (4 * 60 * 60 * 1000)
            temppd['datetime'] = pd.to_datetime(temppd['datetime'], unit='ms')
            temppd.set_index('datetime', inplace=True)
            temppd.to_csv(f"Resources/Data/{s}_daily_data.csv")
            print(s)
            print(temppd)
        except KeyError:
            print(colored(s, "red"))


def main(update, algo, start=datetime(month=4, day=20, year=2016), end=datetime.now()):
    if update:
        ExportDataToCSV(["QQQ",
                         "AAPL", "ADBE", "ADI", "ADSK", "AEP", "ALGN", "AMAT", "AMD",
                         "AMGN", "AMZN", "ANSS", "ASML", "ATVI", "AVGO", "BIDU", "BKNG",
                         "CDNS", "CDW", "CERN", "CHKP", "CMCSA", "COST",
                         "CRWD", "CSCO", "CSX", "CTAS", "CTSH", "DLTR", "DOCU", "DXCM",
                         "EA", "EBAY", "EXC", "FAST", "FB", "FISV", "FOX", "GILD",
                         "GOOG", "HON", "IDXX", "ILMN", "INCY", "INTC", "INTU", "ISRG",
                         "JD", "KDP", "KHC", "KLAC", "LRCX", "LULU", "MAR", "MCHP", "MDLZ",
                         "MELI", "MNST", "MRNA", "MRVL", "MSFT", "MTCH", "MU", "NFLX",
                         "NTES", "NVDA", "NXPI", "OKTA", "ORLY", "PAYX", "PCAR", "PDD",
                         "PEP", "PTON", "PYPL", "QCOM", "REGN", "ROST", "SBUX", "SGEN",
                         "SIRI", "SNPS", "SPLK", "SWKS", "TCOM", "TEAM", "TMUS", "TSLA",
                         "TXN", "VRSK", "VRSN", "VRTX", "WBA", "WDAY", "XEL", "XLNX", "ZM"
                         # "CRPT", "CHTR"
                         ],
                        start, end)

    backtester = Backtest(start, end)
    if algo is None: algo = Algorithm
    algo = algo(start)
    backtester.PreProcess()
    try:
        backtester.RunBacktest(algo)
    except KeyboardInterrupt:
        algo.OnFinish(backtester.QQQ)
    else:
        algo.OnFinish(backtester.QQQ)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-u", "--Update", action='store_true', help="Update CSV Files")
    args = parser.parse_args()

    main(args.Update, None)
