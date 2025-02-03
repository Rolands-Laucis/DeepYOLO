from infer import Model, API
import pandas as pd
import pandas_ta as ta
import numpy as np
import json

col_order = ['Date_sin', 'Open', 'High', 'Low', 'Close', '28_EMA', 'Stoch_RSI', 'Stoch_RSI_D', 'MACD', 'MACD_Signal', 'MACD_Hist', 'SAR', 'SuperTrend']

def proc(data):
    df = pd.DataFrame(columns=['Date', 'Open', 'High', 'Low', 'Close'])
    k = list(data.keys())
    for i in range(len(k)):
        df.loc[len(df)] = [k[i], data[k[i]]['1. open'], data[k[i]]['2. high'], data[k[i]]['3. low'], data[k[i]]['4. close']]
    df['Date'] = pd.to_datetime(df['Date'])
    df.columns = ['Date', 'Open', 'High', 'Low', 'Close']
    df['Open'] = df['Open'].astype(np.float32)
    df['High'] = df['High'].astype(np.float32)
    df['Low'] = df['Low'].astype(np.float32)
    df['Close'] = df['Close'].astype(np.float32)
    # df.set_index(['Date'], inplace=True)
    # print(df.head())

    # indicators
    df.sort_values(['Date'], ascending=True, inplace=True)
    df['28_EMA'] = ta.ema(df['Close'], length=28)
    stoch_rsi = ta.stochrsi(df['Close'], length=14)
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

# m = Model('models\\gru_loss43.736_a0.674_rmse6.613_u128_e10_csvs4000_d21_w8_c5.keras')
# api = API()
# data = api.RequestDaily().get('Time Series (Daily)', {})
with open('resp.json', 'r') as j:
    d = json.load(j)
    x = proc(d)
    # print(x)

    m = Model('models\\gru_loss122.637_a0.723_rmse11.074_u128_e20_csvs11466_d21_w8_c5.keras')
    print('gru')
    print(m.pred(x))
    m = Model('models\\gru_loss43.736_a0.674_rmse6.613_u128_e10_csvs4000_d21_w8_c5.keras')
    print('gru yesterday')
    print(m.pred(x))
    m = Model('models\\lstm_loss62.973_a0.714_rmse7.936_u128_e20_csvs11466_d21_w8_c5.keras')
    print('lstm')
    print(m.pred(x))