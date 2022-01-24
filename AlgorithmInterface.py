import pandas as pd
import matplotlib.pyplot as plt
import traceback
import math
import numpy as np


class Wallet:
    def __init__(self, start, bal=100000, track_extra=None, reset_balance=False):
        if track_extra is None: track_extra = []
        self.reset_balance = reset_balance

        self.Starting_Balance = bal
        self.Balance = bal
        self.PrevBalance = self.Balance

        self.TradeLog = {} # pd.DataFrame(columns=['Type', 'Ticker', 'Price', 'Volume', 'Datetime'])
        self.TradeLogColumns = ['Type', 'Ticker', 'Price', 'Volume', 'Datetime']
        self.CurrentTrades = {}

        self.IN_TRADE_TICKER_LONG = []
        self.IN_TRADE_TICKER_SHORT = []
        self.IN_TRADE = False

        self.TrackData = {start: [self.Balance, 1, None] + [None] * len(track_extra)}
        self.TrackDataColumns = ['Balance', 'pChange', 'Type'] + track_extra
        # pd.DataFrame(columns=data_columns)

    def PlaceBuyOrder(self, ticker, price, date, split=1, extra_data=None):
        if extra_data is None: extra_data = []

        if price < 0:
            print("Something went wrong!")
            print(ticker)

        volume = math.floor((self.Balance / price) / split)

        self.Balance -= price * volume
        self.TradeLog[date] = ['BUY', ticker, price, volume, date]

        self.CurrentTrades[ticker] = ('BUY', price, volume, extra_data)
        self.IN_TRADE_TICKER_LONG.append(ticker)
        self.IN_TRADE = True

    def PlaceShortOrder(self, ticker, price, date, split=2, extra_data=None):
        if extra_data is None: extra_data = []

        if price < 0:
            print("Something went wrong!")
            print(ticker)

        volume = math.floor((self.Balance / price) / split)

        self.Balance += price * volume
        self.TradeLog[date] = ['SHORT', ticker, price, volume, date]

        self.CurrentTrades[ticker] = ('SHORT', price, volume, extra_data)
        self.IN_TRADE_TICKER_SHORT.append(ticker)
        self.IN_TRADE = True

    def CloseOrder(self, ticker, price, date, extra_data_close=None):
        if extra_data_close is None: extra_data_close = []

        trade_type, prev_price, volume, extra_data = self.CurrentTrades.pop(ticker)

        if trade_type == 'BUY':
            self.Balance += (price * volume)
            self.TrackData[date] = [self.Balance, (self.Balance / self.PrevBalance) - 1, trade_type] + extra_data + extra_data_close
            # self.IN_TRADE_TICKER_LONG.remove(ticker)

        if trade_type == 'SHORT':
            self.Balance -= (price * volume)
            self.TrackData[date] = [self.Balance, (self.Balance / self.PrevBalance) - 1, trade_type] + extra_data + extra_data_close
            # self.IN_TRADE_TICKER_SHORT.remove(ticker)

        # if self.reset_balance: self.Balance = self.Starting_Balance
        self.TradeLog[date] = [f'CLOSE-{trade_type}', ticker, price, volume, date]

    def CloseAllOrders(self, currentSlice, idx, extra_data_close=None):
        if extra_data_close is None: extra_data_close = []

        if self.IN_TRADE:
            if self.IN_TRADE_TICKER_LONG:
                for ticker in self.IN_TRADE_TICKER_LONG:
                    close_long = currentSlice[ticker].close
                    self.CloseOrder(ticker, close_long, idx, extra_data_close)

            if self.IN_TRADE_TICKER_SHORT:
                for ticker in self.IN_TRADE_TICKER_SHORT:
                    close_short = currentSlice[ticker].close
                    self.CloseOrder(ticker, close_short, idx, extra_data_close)

            self.IN_TRADE_TICKER_LONG = []
            self.IN_TRADE_TICKER_SHORT = []
            self.IN_TRADE = False

    def PlotData(self, QQQ):

        self.TradeLog = pd.DataFrame.from_dict(self.TradeLog, orient='index', columns=self.TradeLogColumns)
        self.TrackData = pd.DataFrame.from_dict(self.TrackData, orient='index', columns=self.TrackDataColumns)
        print(self.TradeLog)
        print(self.TrackData)

        self.TradeLog.to_csv("Resources/Output/Trade_Log_recent.csv")
        self.TrackData.to_csv("Resources/Output/Track_Data_recent.csv")

        self.TrackData['log_balance'] = np.log(self.TrackData['Balance'])
        QQQ['Rolling_pChange'] = QQQ['pChange'].rolling(len(QQQ.index), min_periods=1).apply(np.prod)
        QQQ['Balance'] = self.Starting_Balance * QQQ['Rolling_pChange']
        QQQ['log_balance'] = np.log(QQQ['Balance'])

        fig, ax1 = plt.subplots()

        ax1.plot(self.TrackData.index, self.TrackData.log_balance, c='green', label='Algorithm')
        ax1.plot(QQQ.index, QQQ.log_balance, c='blue', label='QQQ')

        plt.legend()
        plt.show()


class Algorithm:
    def __init__(self, start):
        self.tickerData = {}
        self.currentSlice = {}
        self.now = None
        self.wallet = Wallet(start)

    def OnData(self):
        pass

    def OnFinish(self, market):
        pass
