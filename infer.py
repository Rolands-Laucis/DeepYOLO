class Model:
    def __init__(self, path=str):
        import numpy as np
        import pandas as pd
        from tensorflow.keras.models import load_model
        self.np = np
        self.pd = pd

        self.model = load_model(path)
        self.model.trainable = False  # Ensures the model is not accidentally retrained

    def PrepData(self, api_obj:dict):
        df = self.pd.DataFrame.from_dict(api_obj, orient="index")
        df.rename(columns={"index": "date"}, inplace=True)
        df['date'] = self.pd.to_datetime(df['date'])

        last_day = df['date'].max()
        days = df[df['date'] >= (last_day - self.pd.Timedelta(days=21))]

        weeks = df[df['date'] >= (last_day - self.pd.Timedelta(weeks=8))]
        weeks = df.resample('W-FRI').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last'
        })
        return [self.np.array(days), self.np.array(weeks)]

    def pred(self, x=[]):
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
        url = self.api_base_url + self.QueryParams({
            "symbol": sym,
            "outputsize":"compact",
            "extended_hours": "False"
        } | params | self.api_key_param)
        r = self.requests.get(url)
        return r.json()

    def RequestIntraday(self, sym:str='AAPL', interval='60min') -> dict:
        return self.Request(sym, {"function": "TIME_SERIES_INTRADAY", "interval": interval})

    def RequestDaily(self, sym:str='AAPL') -> dict:
        return self.Request(sym, {"function": "TIME_SERIES_DAILY"}).get('Time Series (Daily)', {})