from dotenv import dotenv_values
import requests
from pprint import pprint

# load vars from local .env file
env = dotenv_values(".env")
api_key = env.get("Alpha_Vantage", "demo")
api_key_param = {"apikey": api_key}

def QueryParams(d:dict) -> str:
    return '?' + "&".join([f"{k}={v}" for k, v in d.items()])

# alphavantage API documentation: https://www.alphavantage.co/documentation/
api_base_url = "https://www.alphavantage.co/query"
def Request(sym:str='AAPL', interval='60min') -> dict:
    url = api_base_url + QueryParams({
        "function": "TIME_SERIES_DAILY",
        "symbol": sym,
        "interval": interval,
        "extended_hours": "False",
    } | api_key_param)

    r = requests.get(url)
    return r.json()

# pprint(Request())