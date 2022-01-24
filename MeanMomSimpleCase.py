import GenericBacktester
import argparse
from AlgorithmInterface import Algorithm, Wallet
import pandas as pd
import statistics
from datetime import datetime
import random
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import math


class MM_SimpleCase(Algorithm):
    def __init__(self, start):
        super().__init__(start)
        # self.tickerData = {}
        # self.currentSlice = {}
        # self.now = None
        self.wallet = Wallet(start, track_extra=['qqq_pChange'])

        self.test_wallets = {}
        for i in ['mean_long', 'mom_long', 'mean_short', 'mom_short']:
            self.test_wallets[i] = Wallet(start, track_extra=None, reset_balance=False)

        self.PrevRanking = {}
        self.TrackRanking = {}
        for i in range(0, 83):
            self.TrackRanking[i] = []

        self.trackingIndex = 0

        self.lookback_ranking = 5
        self.lookback_returns = 10

    def OnData(self):
        qqq = self.currentSlice['QQQ']

        for i, test_wallet in self.test_wallets.items():
            if test_wallet.IN_TRADE: test_wallet.CloseAllOrders(self.currentSlice, self.now)
            test_wallet.PrevBalance = test_wallet.Balance

        if self.wallet.IN_TRADE: self.wallet.CloseAllOrders(self.currentSlice, self.now, [qqq.pChange])
        self.wallet.PrevBalance = self.wallet.Balance

        ranking_df = self.Ranking()
       # self.UpdateTrackRanking(ranking_df)

        prev_returns = self.PreviousReturns()

        best_strat = max(prev_returns, key=prev_returns.get)

        min_ticker = ranking_df.iloc[0]
        max_ticker = ranking_df.iloc[-1]

        self.test_wallets['mean_long'].PlaceBuyOrder(min_ticker.name, min_ticker.close, self.now, split=1)
        self.test_wallets['mom_long'].PlaceBuyOrder(max_ticker.name, max_ticker.close, self.now, split=1)
        self.test_wallets['mom_short'].PlaceShortOrder(min_ticker.name, min_ticker.close, self.now, split=3)
        self.test_wallets['mean_short'].PlaceShortOrder(max_ticker.name, max_ticker.close, self.now, split=3)

        if self.trackingIndex > self.lookback_returns:
            if 'mean' in best_strat:
                if 'long' in best_strat:
                    self.wallet.PlaceBuyOrder(min_ticker.name, min_ticker.close, self.now, split=1)
                if 'short' in best_strat:
                    self.wallet.PlaceShortOrder(max_ticker.name, max_ticker.close, self.now, split=3)
            if 'mom' in best_strat:
                if 'long' in best_strat:
                    self.wallet.PlaceBuyOrder(max_ticker.name, max_ticker.close, self.now, split=1)
                if 'short' in best_strat:
                    self.wallet.PlaceShortOrder(min_ticker.name, min_ticker.close, self.now, split=3)

        self.trackingIndex += 1

    def UpdateTrackRanking(self, ranking_df):
        i = 0
        for idx, row in ranking_df.iterrows():
            try:
                prev_ranking = self.PrevRanking[idx]
            except KeyError:
                self.PrevRanking[idx] = i
                i += 1
                continue

            try:
                self.TrackRanking[prev_ranking].append(i)
            except KeyError:
                print(prev_ranking)
                quit(1)
                self.TrackRanking[prev_ranking] = [i]

            self.PrevRanking[idx] = i

            i += 1

    def TrackRankingData(self):
        df = pd.DataFrame.from_dict(self.TrackRanking, orient='index').transpose()
        hist = df[[79, 80, 81, 82]].hist(bins=len(df.columns))
        plt.show()

    def OnFinish(self, QQQ):
        QQQ = QQQ.loc[:self.now]
        # self.TrackRankingData()
        self.wallet.PlotData(QQQ)
        self.CustomPlot(QQQ)

    def CustomPlot(self, QQQ):
        figure, axis = plt.subplots(2, 2)

        QQQ['Rolling_pChange'] = QQQ['pChange'].rolling(len(QQQ.index), min_periods=1).apply(np.prod)
        QQQ['Balance'] = self.wallet.Starting_Balance * QQQ['Rolling_pChange']
        QQQ['log_balance'] = np.log(QQQ['Balance'])

        mean_long = self.test_wallets['mean_long']
        mean_short = self.test_wallets['mean_short']
        mom_long = self.test_wallets['mom_long']
        mom_short = self.test_wallets['mom_short']

        mean_long.TrackData = pd.DataFrame.from_dict(mean_long.TrackData,
                                                     orient='index', columns=mean_long.TrackDataColumns)
        mean_long.TrackData['log_balance'] = np.log(mean_long.TrackData['Balance'])

        mean_short.TrackData = pd.DataFrame.from_dict(mean_short.TrackData,
                                                     orient='index', columns=mean_short.TrackDataColumns)
        mean_short.TrackData['log_balance'] = np.log(mean_short.TrackData['Balance'])

        mom_long.TrackData = pd.DataFrame.from_dict(mom_long.TrackData,
                                                     orient='index', columns=mom_long.TrackDataColumns)
        mom_long.TrackData['log_balance'] = np.log(mom_long.TrackData['Balance'])

        mom_short.TrackData = pd.DataFrame.from_dict(mom_short.TrackData,
                                                     orient='index', columns=mean_long.TrackDataColumns)
        mom_short.TrackData['log_balance'] = np.log(mom_short.TrackData['Balance'])

        axis[0, 0].plot(mean_long.TrackData.index, mean_long.TrackData.log_balance, c='green', label='Algorithm')
        axis[0, 0].plot(QQQ.index, QQQ.log_balance, c='blue', label='QQQ')
        axis[0, 0].title.set_text("Mean Reversion Long")

        axis[1, 0].plot(mean_short.TrackData.index, mean_short.TrackData.log_balance, c='green', label='Algorithm')
        axis[1, 0].plot(QQQ.index, QQQ.log_balance, c='blue', label='QQQ')
        axis[1, 0].title.set_text("Mean Reversion Short")

        axis[0, 1].plot(mom_long.TrackData.index, mom_long.TrackData.log_balance, c='green', label='Algorithm')
        axis[0, 1].plot(QQQ.index, QQQ.log_balance, c='blue', label='QQQ')
        axis[0, 1].title.set_text("Momentum Long")

        axis[1, 1].plot(mom_short.TrackData.index, mom_short.TrackData.log_balance, c='green', label='Algorithm')
        axis[1, 1].plot(QQQ.index, QQQ.log_balance, c='blue', label='QQQ')
        axis[1, 1].title.set_text("Momentum Short")

        plt.legend()
        plt.show()

    def PreviousReturns(self):
        returns = {}
        for i, wallet in self.test_wallets.items():
            temp_returns = pd.DataFrame.from_dict(wallet.TrackData, orient='index',
                                                  columns=['balance', 'pChange', 'type']).iloc[-(self.lookback_returns + 1):]

            temp_returns['pChange'] = (temp_returns['balance'] / temp_returns['balance'].shift(1)) - 1

            temp_returns.dropna(inplace=True)

            returns[i] = temp_returns['pChange'].median()

        return returns

    def Ranking(self):
        ranking_dict = {}
        for ticker, data in self.tickerData.items():
            recent_data = data.iloc[-1]

            ranking_dict[ticker] = [recent_data['pChange'], recent_data['close']]

        ranking_df = pd.DataFrame.from_dict(ranking_dict, orient='index', columns=['score', 'close'])
        ranking_df = ranking_df.sort_values(by='score')

        return ranking_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-u", "--Update", action='store_true', help="Update CSV Files")
    args = parser.parse_args()

    GenericBacktester.main(args.Update, MM_SimpleCase,
                           start=datetime(month=1, day=1, year=2014),
                           end=datetime(month=12, day=31, year=2018))
