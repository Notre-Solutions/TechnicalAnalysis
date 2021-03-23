import pandas as pd 
from backtesting import Strategy
from backtesting.lib import crossover
import seaborn as sns
import plotly.express as px
from copy import copy
from scipy import stats
import pickle
import matplotlib.pyplot as plt
import numpy as np
import plotly.figure_factory as ff
import plotly.graph_objects as go
import ta

class NotreTA():

    def cleanData(self, df, indexColName='Date', droppedCols=['Unnamed: 0']):
        df = df.dropna()
        df.set_index(indexColName, inplace=True)
        df.index = pd.to_datetime(df.index)
        if 'Unnamed: 0' in df.columns:
            df = df.drop(columns=droppedCols)
        df.columns = map(str.capitalize, df.columns)
        return df

    def saveStrategy(self, stats):
        name = str(stats._strategy).split('(')[0]
        rbhRatio = stats['Return [%]'] / stats['Buy & Hold Return [%]']
        today = datetime.now().strftime("%d_%m_%Y")
        fileName = f'/content/drive/My Drive/Colab Notebooks/Strategies/{today}_{name}_{rbhRatio}.pickle'

        with open(fileName, 'wb') as handle:
            pickle.dump(stats, handle, protocol=pickle.HIGHEST_PROTOCOL)

        def getStrategy(self, fileName):
        with open(f'/content/drive/My Drive/Colab Notebooks/Strategies/{fileName}', 'rb') as handle:
            b = pickle.load(handle)
        return b

    def expectancy(self, stats):
        t = stats._trades
        pos = t[t['PnL']>0]
        posSize = sum(pos['PnL'])
        neg = t[t['PnL']<0]
        negSize = abs(sum(neg['PnL']))
        factor = stats['Win Rate [%]']/100
        expectancy = (1 + posSize/negSize) * factor - 1
        print(f'1 plus (Pos Size: {posSize} DIV BY Neg Size: {negSize}) TIMES Win Rate: {factor} minus 1')
        return expectancy

    def SMA(self, values, n):
        """
        Return simple moving average of `values`, at
        each step taking into account `n` previous values.
        """
        return pd.Series(values).rolling(n).mean()

    def DMI(self, data, window=14):
        df = data.copy()
        adx = ta.trend.ADXIndicator(df.High,df.Low,df.Close,window)
        df['dm_neg'] = adx.adx_neg()
        df['dm_pos'] = adx.adx_pos()
        df['dm_adx'] = adx.adx()
        return df['dm_neg'], df['dm_pos'], df['dm_adx'] 

    def interactive_plot(self, df, title):
        fig = px.line(title=title)

        for i in df.columns:
            fig.add_scatter(x=df.index, y=df[i], name=i)
        
        fig.show()
    
    def getData(self, base, quote, interval, resample):

        filePath = f'/content/drive/My Drive/Colab Notebooks/Crypto/{base}-{quote}-{interval}.csv'

        data = pd.read_csv(filePath)
        data.open_time = [datetime.utcfromtimestamp(i/1000).strftime('%Y-%m-%d %H:%M:%S') for i in data.open_time]
        data = cleanData(data, 'open_time', droppedCols=['Unnamed: 0','quote_asset_volume',	'number_of_trades',	'taker_buy_base_asset_volume',	'taker_buy_quote_asset_volume',	'ignore','close_time'])
        data
        data_interval = (data.resample(resample)
                        .agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'}))
        data_interval = data_interval[1:]
        return data_interval

        import warnings
        warnings.filterwarnings("ignore")

    def walkForwardAnalysis(self, data, trainMonths, testMonths ):
        x = data.index
        monthsInSet = (max(x) - min(x) )/np.timedelta64(1, 'M')
        if monthsInSet < (testMonths + trainMonths):
            return False
        else:
            notEnd = True
            parameters = {}
            start = min(x) + np.timedelta64(trainMonths,'M')
            end = max(x)
            while notEnd:
            trainStart = start - np.timedelta64(trainMonths,'M')
            train = data[trainStart:start.strftime("%Y-%m-%d")]
            intermediate = start+np.timedelta64(testMonths,'M')
            if intermediate < end:
                test = data[start.strftime("%Y-%m-%d"):intermediate.strftime("%Y-%m-%d")]
            else:
                test = data[start.strftime("%Y-%m-%d"):end.strftime("%Y-%m-%d")]

                notEnd = False
            
            print('Train:',train.shape)
            print(min(train.index), max(train.index))
            print('Test:',test.shape)
            print(min(test.index), max(test.index))
            start = intermediate

            bTrain = Backtest(train.interpolate(), DMI_STRAT, cash=100, commission=.005)
            
            stats = bTrain.optimize(
                                    dmi_window=range(49, 50, 1),
                                    loss_percentage=range(19, 20, 1),
                                    adx_threshold=range(0,1,1),
                                    maximize='Equity Final [$]'
                                    )
            
            print()
            parameters = dict(dmi_window=stats._strategy.dmi_window,
                                loss_percentage=stats._strategy.loss_percentage,
                                adx_threshold=stats._strategy.adx_threshold)
            print()
            print('Paramters',str(parameters))
            bTest = Backtest(test.interpolate(), DMI_STRAT, cash=1000000, commission=.005)
            stats = bTest.run(**parameters)
            try:
                print('First expectancy: ', expectancy(stats))
                t = stats._trades
                avgWin = np.mean(t[t['PnL']>0]['PnL'])
                avgLoss = np.mean(t[t['PnL']<0]['PnL'])
                avgLoss
                expec = (stats['Win Rate [%]']/100)*avgWin - (1 - (stats['Win Rate [%]']/100))*abs(avgLoss)
                
                print(f'Expectancy: {expec}')
            except Exception as e:
                print(e)
            print(stats)
            print("-------------------")



if __name__="__main__":
    print("Please import and create NotreTA Class to use methods")