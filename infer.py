import numpy as np
import pandas as pd
import pandas_ta as ta

timestamp_2000 = pd.Timestamp('2000-01-01')

class Model:
    def __init__(self, name:str=''):
        print('Importing model libs...')
        from tensorflow.keras.models import load_model
        import json
        from pprint import pprint

        self.load_model = load_model
        self.json = json
        self.pp = pprint
        self.name = name

    def LoadModel(self, json_path:str, name:str=''):
        print('Loading config...')
        if name: self.name = name

        with open(json_path, 'r') as p:
            self.config = self.json.load(p)
        model_path = self.config.get('model_path', '')
        print(f'Loading model...', model_path)
        self.model = self.load_model(model_path)
        self.model.trainable = False  # Ensures the model is not accidentally retrained

        print('Loaded model:')
        # self.pp({key: self.config[key] for key in ['rnn_type', 'rnn_units', 'total_csvs', 'epochs', 'total_csvs', 'metrics']})

    def PrepData(self, api_obj:dict) -> list:
        cols = ['Date', 'Open', 'Close', 'High', 'Low']
        df = pd.DataFrame(columns=cols)
        k = list(api_obj.keys())
        for i in range(len(k)):
            df.loc[len(df)] = [k[i], api_obj[k[i]]['1. open'], api_obj[k[i]]['4. close'], api_obj[k[i]]['2. high'], api_obj[k[i]]['3. low']]
        df['Date'] = pd.to_datetime(df['Date'])
        df.columns = cols
        df['Open'] = df['Open'].astype(np.float32)
        df['High'] = df['High'].astype(np.float32)
        df['Low'] = df['Low'].astype(np.float32)
        df['Close'] = df['Close'].astype(np.float32)
        # print(df.head())

        # indicators
        df.sort_values(['Date'], ascending=True, inplace=True)
        df['EMA'] = ta.ema(df['Close'], length=21)

        stoch_rsi = ta.stochrsi(df['Close'])
        df['Stoch_RSI'] = stoch_rsi.iloc[:, 0]
        df['Stoch_RSI_D'] = stoch_rsi.iloc[:, 1]

        macd = ta.macd(df['Close'])#.dropna()
        df['MACD'] = macd.iloc[:, 0]
        df['MACD_Signal'] = macd.iloc[:, 2]

        df['SAR'] = ta.psar(df['High'], df['Low'], df['Close'])['PSARl_0.02_0.2']
        df['SuperTrend'] = ta.supertrend(df['High'], df['Low'], df['Close'])['SUPERT_7_3.0']

        ## Overbought/Oversold Indicators
        bb = ta.bbands(df['Close'], length=2) # Bollinger Bands
        df['BOLL_Low'] = bb.iloc[:, 0]
        df['BOLL_High'] = bb.iloc[:, 2]

        ## Energy Indicators
        df['PSL'] = ta.psl(df['Close'])  # Psychological Line (default length = 12)
        df['MASS'] = ta.massi(df['High'], df['Low'])  # Mass Index (default settings)

        ## Volatility Indicators
        df['NATR'] = ta.natr(df['High'], df['Low'], df['Close'])  # Normalized ATR

        ## Momentum Indicators
        df['ROC'] = ta.roc(df['Close'])  # Rate of Change
        df['WILLR'] = ta.willr(df['High'], df['Low'], df['Close'])  # Williams %R

        dmi = ta.adx(df['High'], df['Low'], df['Close']).dropna()
        df['DMI_ADX'] = dmi.iloc[:, 0]
        df['DMI_DI_P'] = dmi.iloc[:, 1]
        df['DMI_DI_N'] = dmi.iloc[:, 2]

        # Convert the date column to a cyclical year 1-364 day representation
        df['Day'] = df['Date'].dt.dayofyear
        df['Date_sin'] = np.sin(2 * np.pi * df['Day'] / 365)
        df['Date_cos'] = np.cos(2 * np.pi * df['Day'] / 365)
        df.drop(columns=['Day'], inplace=True)
        df.sort_values(['Date'], ascending=False, inplace=True)
        # print(df.head())
        # print(df.shape)
        # exit(0)
        
        feats = set(df.columns) - set(['Date', 'Date_sin', 'Date_cos', 'Open', 'Close', 'High', 'Low'])
        feat_dict = dict(map(lambda f: (f, 'mean'), feats))
        df.set_index('Date', inplace=True)  # Ensure DateTimeIndex
        weeks = df.iloc[:self.config['lookback_weeks']*7].copy()
        weeks = weeks.resample('W-FRI').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last'
        } | feat_dict).reset_index()
        weeks.sort_values(['Date'], ascending=False, inplace=True)
        weeks.reset_index(inplace=True)
        weeks['Week'] = weeks['Date'].dt.isocalendar().week
        weeks['Date_sin'] = np.sin(2 * np.pi * weeks['Week'] / 52)
        weeks['Date_cos'] = np.cos(2 * np.pi * weeks['Week'] / 52)
        weeks.drop(['Week', 'Date', 'index'], axis=1, inplace=True)
        weeks = weeks[self.config['columns']]
        weeks = weeks.iloc[:self.config['lookback_weeks']]
        weeks = weeks.fillna(0).astype('float32')
        # print(weeks.head())
        # print(weeks.shape, type(weeks))
        # exit(0)

        # Resample to monthly frequency, considering business days
        months = df.iloc[:self.config['lookback_months']*31].copy()
        months = months.resample('ME').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last'
        } | feat_dict).dropna()
        months.reset_index(inplace=True)
        
        # Convert the date column to a cyclical year 1-12 month representation
        months['Month'] = months['Date'].dt.month
        months['Date_sin'] = np.sin(2 * np.pi * months['Month'] / 12)
        months['Date_cos'] = np.cos(2 * np.pi * months['Month'] / 12)
        months.drop(columns=['Month'], inplace=True)
        months = months.iloc[:self.config['lookback_months']]
        months = months[self.config['columns']]
        months = months.fillna(0).astype('float32')
        # print(months.head())
        # print(months.shape)
        # exit(0)
        
        df = df[:self.config['lookback_days']]
        df = df[self.config['columns']]
        df = df.fillna(0).astype('float32')
        df = df.values.reshape((1, self.config['lookback_days'], len(self.config['columns'])))
        weeks = weeks.values.reshape((1, self.config['lookback_weeks'], len(self.config['columns'])))
        months = months.values.reshape((1, self.config['lookback_months'], len(self.config['columns'])))
        return [df, weeks, months]

    def pred(self, x=[], ticker:str=''):
        print(f'{self.name} predicting {ticker}...')
        return self.model.predict(x)[0]

class API:
    # alphavantage API documentation: https://www.alphavantage.co/documentation/
    api_base_url = "https://www.alphavantage.co/query"
    commodity_syms = {
        'OIL': 'WTI',
        'WTI': 'WTI',
        'GOLD': 'GOLD',
        'BRENT': 'BRENT',
        'NATGAS': 'NATURAL_GAS',
        'SILVER': 'SILVER',
        'COPPER': 'COPPER',
    }

    def __init__(self):
        import requests
        from dotenv import dotenv_values
        self.requests = requests

        # load vars from local .env file
        env = dotenv_values(".env")
        api_key = env.get("Alpha_Vantage", "demo")
        self.api_key_param = {"apikey": api_key}

    def QueryParams(self, d:dict) -> str:
        return '?' + "&".join([f"{k}={v}" for k, v in d.items()])

    def Request(self, sym:str, params:dict={}):
        print(f'Requesting {sym} ...')
        if sym in self.commodity_syms:
            params['function'] = self.commodity_syms[sym]
            sym = ''
        params = {
            "outputsize":"compact",
            "extended_hours": "False"
        } | params | self.api_key_param
        if sym:
            params["symbol"] = sym
        url = self.api_base_url + self.QueryParams(params)
        r = self.requests.get(url)
        print(f'HTTP {r.status_code}')
        if r.status_code != 200:
            raise ValueError('Request failed HTTP', r.status_code, r.text)
        j = r.json()
        err = j.get('Error Message', None)
        if err:
            raise ValueError('Request error msg', r.status_code, r.reason,err)
        return j

    def RequestIntraday(self, sym:str='AAPL', interval='60min') -> dict:
        return self.Request(sym, {"function": "TIME_SERIES_INTRADAY", "interval": interval})

    def RequestDaily(self, sym:str='AAPL') -> dict:
        r = self.Request(sym, {"function": "TIME_SERIES_DAILY"})
        r = r.get('Time Series (Daily)', {})
        if not r:
            raise ValueError('Request response empty', r)
        return r
    
    # def RequestCommodity(self, func:str='OIL') -> dict:
    #     if func in self.commodity_funcs:
    #         func = self.commodity_funcs[func]
    #     print(f'Requesting commodity {func} ...')
    #     return self.Request('', {"function": func, "datatype": "json"})
    

# col_order = ['Date_sin', 'Open', 'High', 'Low', 'Close', '28_EMA', 'Stoch_RSI', 'Stoch_RSI_D', 'MACD', 'MACD_Signal', 'MACD_Hist', 'SAR', 'SuperTrend']
