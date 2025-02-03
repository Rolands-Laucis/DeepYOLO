import numpy as np
import pandas as pd
import pandas_ta as ta

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
        self.pp({key: self.config[key] for key in ['rnn_type', 'rnn_units', 'total_csvs', 'epochs', 'total_csvs', 'metrics']})

    def pred(self, x=[]):
        print(f'{self.name} Predicting...')
        return self.model.predict(x)[0]

class API:
    # alphavantage API documentation: https://www.alphavantage.co/documentation/
    api_base_url = "https://www.alphavantage.co/query"

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
        url = self.api_base_url + self.QueryParams({
            "symbol": sym,
            "outputsize":"compact",
            "extended_hours": "False"
        } | params | self.api_key_param)
        r = self.requests.get(url)
        print(f'HTTP {r.status_code} Got response')
        if r.status_code != 200:
            raise ValueError('Request failed HTTP', r.status_code, r.text)
        j = r.json()
        err = j.get('Error Message', None)
        if err:
            raise ValueError('Request error msg', r.status_code, err)
        return j

    def RequestIntraday(self, sym:str='AAPL', interval='60min') -> dict:
        return self.Request(sym, {"function": "TIME_SERIES_INTRADAY", "interval": interval})

    def RequestDaily(self, sym:str='AAPL') -> dict:
        r = self.Request(sym, {"function": "TIME_SERIES_DAILY"})
        r = r.get('Time Series (Daily)', {})
        if not r:
            raise ValueError('Request response empty', r)
        return r
    

col_order = ['Date_sin', 'Open', 'High', 'Low', 'Close', '28_EMA', 'Stoch_RSI', 'Stoch_RSI_D', 'MACD', 'MACD_Signal', 'MACD_Hist', 'SAR', 'SuperTrend']
def PrepData(api_obj:dict):
    global col_order

    df = pd.DataFrame(columns=['Date', 'Open', 'High', 'Low', 'Close'])
    k = list(api_obj.keys())
    for i in range(len(k)):
        df.loc[len(df)] = [k[i], api_obj[k[i]]['1. open'], api_obj[k[i]]['2. high'], api_obj[k[i]]['3. low'], api_obj[k[i]]['4. close']]
    df['Date'] = pd.to_datetime(df['Date'])
    df.columns = ['Date', 'Open', 'High', 'Low', 'Close']
    df['Open'] = df['Open'].astype(np.float32)
    df['High'] = df['High'].astype(np.float32)
    df['Low'] = df['Low'].astype(np.float32)
    df['Close'] = df['Close'].astype(np.float32)
    # df.set_index(['Date'], inplace=True)
    print(df.head())

    # indicators
    df.sort_values(['Date'], ascending=True, inplace=True)
    df['28_EMA'] = ta.ema(df['Close'], length=28)
    stoch_rsi = ta.stochrsi(df['Close'], length=14)
    if stoch_rsi is None:
        print(df.head())
    df['Stoch_RSI'] = stoch_rsi['STOCHRSIk_14_14_3_3']
    df['Stoch_RSI_D'] = stoch_rsi['STOCHRSId_14_14_3_3']
    macd = ta.macd(df['Close'])
    df['MACD'] = macd['MACD_12_26_9']
    df['MACD_Signal'] = macd['MACDs_12_26_9']
    df['MACD_Hist'] = macd['MACDh_12_26_9']
    df['SAR'] = ta.psar(df['High'], df['Low'], df['Close'])['PSARl_0.02_0.2']
    df['SuperTrend'] = ta.supertrend(df['High'], df['Low'], df['Close'])['SUPERT_7_3.0']
    df.sort_values(['Date'], ascending=False, inplace=True)
    # print(df.head())

    # time scales
    days = df.copy().iloc[:21]
    days['Date'] = days['Date'].dt.dayofyear
    days['Date_sin'] = np.sin(2 * np.pi * days['Date'] / 365)
    days.drop('Date', axis=1, inplace=True)
    days = days[col_order]
    days = days.fillna(0).astype('float32')
    # print(days.head())
    # print(days.shape)
    # exit(0)

    df.set_index('Date', inplace=True)
    weeks = df.iloc[:8*7]
    # print(weeks.dtypes, weeks.columns)
    weeks = weeks.resample('W-FRI').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last', 
        '28_EMA':'mean', 
        'Stoch_RSI':'mean', 
        'Stoch_RSI_D':'mean', 
        'MACD':'mean', 
        'MACD_Signal':'mean', 
        'MACD_Hist':'mean', 
        'SAR':'mean', 
        'SuperTrend':'mean'
    }).reset_index()
    weeks.sort_values(['Date'], ascending=False, inplace=True)
    weeks.reset_index(inplace=True)
    weeks['Week'] = weeks['Date'].dt.isocalendar().week
    weeks['Date_sin'] = np.sin(2 * np.pi * weeks['Week'] / 52)
    weeks.drop(['Week', 'Date', 'index'], axis=1, inplace=True)
    weeks = weeks[col_order]
    weeks = weeks.iloc[:8]
    weeks = weeks.fillna(0).astype('float32')
    # print(weeks.head())
    # print(weeks.shape)
    
    days = days.values.reshape((1, 21, 13)) #np.array([1, ])
    weeks = weeks.values.reshape((1, 8, 13)) #np.array([1, ])
    # weeks = np.array([1, weeks.values])
    return [days, weeks]