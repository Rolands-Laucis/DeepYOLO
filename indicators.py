# example csv:
    # Open,High,Low,Close,Week_sin
    # 1368.692993,1522.824951,1361.213989,1516.604004,0.12053668025532305
    # 1531.712036,1547.708008,1401.706055,1408.848022,0.23931566428755774
    # 1436.887939,1476.456055,1421.641968,1465.084961,0.3546048870425356

# and in new columns calculates the following indicators:
# 50-day moving average
# stohastic RSI
# MACD
# Parabolic SAR
# SuperTrend

# If you encounter a ModuleNotFoundError for 'pkg_resources', install setuptools:
# pip install setuptools


import pandas_ta as ta
import pandas as pd
from glob import glob
import os
import numpy as np
import argparse
import traceback

print("Running indicators.py...")

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--csvs', type=int, help='End number for the glob index filter')
args = parser.parse_args()

timestamp_2000 = pd.Timestamp('2000-01-01')
scales = ['days', 'weeks', 'months'] #
processed_files = {scale: set() for scale in scales}

for j, time_scale in enumerate(scales):
    csvs = glob(f'data/{time_scale}/*.csv')
    if args.csvs:
        csvs = csvs[:args.csvs]
    count = len(csvs)
    print(f"\nProcessing {time_scale}... {count} csvs")
    for i, csv in enumerate(csvs):
        file_id = os.path.basename(csv)
        if j > 0 and file_id not in processed_files[scales[j-1]]:
            continue  # Skip processing if the file was not processed in the previous scale

        df = pd.read_csv(csv)
        try:
            df['EMA'] = ta.ema(df['Close'], length=14)

            stoch_rsi = ta.stochrsi(df['Close'], length=14)
            df['Stoch_RSI'] = stoch_rsi['STOCHRSIk_14_14_3_3']
            df['Stoch_RSI_D'] = stoch_rsi['STOCHRSId_14_14_3_3']

            macd = ta.macd(df['Close'])
            df['MACD'] = macd['MACD_12_26_9']
            df['MACD_Signal'] = macd['MACDs_12_26_9']
            # df['MACD_Hist'] = macd['MACDh_12_26_9']

            df['SAR'] = ta.psar(df['High'], df['Low'], df['Close'])['PSARl_0.02_0.2']
            df['SuperTrend'] = ta.supertrend(df['High'], df['Low'], df['Close'])['SUPERT_7_3.0']

            ## Overbought/Oversold Indicators
            bb = ta.bbands(df['Close'], length=2) # Bollinger Bands
            df['BOLL_Low'] = bb.iloc[:, 0]
            # df['BOLL_Mid'] = bb.iloc[:, 1]
            df['BOLL_High'] = bb.iloc[:, 2]
            # df['WR_WR2'] = ta.wr(df['High'], df['Low'], df['Close'], length=14)  # Williams %R with default length

            ## Energy Indicators
            df['PSL'] = ta.psl(df['Close'], length=14)  # Psychological Line (default length = 12)
            df['MASS'] = ta.massi(df['High'], df['Low'])  # Mass Index (default settings)

            ## Volatility Indicators
            df['NATR'] = ta.natr(df['High'], df['Low'], df['Close'], length=14)  # Normalized ATR
            # df['TRANGE'] = ta.trange(df['High'], df['Low'], df['Close'])  # True Range

            ## Momentum Indicators
            df['ROC'] = ta.roc(df['Close'], length=14)  # Rate of Change (default length = 12)
            df['WILLR'] = ta.willr(df['High'], df['Low'], df['Close'], length=14)  # Williams %R

            dmi = ta.adx(df['High'], df['Low'], df['Close'], length=14)
            df['DMI_ADX'] = dmi['ADX_14']
            df['DMI_DI_P'] = dmi['DMP_14']
            df['DMI_DI_N'] = dmi['DMN_14']

            ## Hilbert Transform - Dominant Cycle Period and Phase
            # df['HT_DCPERIOD'] = ta.ht_dcperiod(df['Close'])  # Dominant Cycle Period
            # df['HT_DCPHASE'] = ta.ht_dcphase(df['Close'])    # Dominant Cycle Phase

            brar = ta.aroon(df['High'], df['Low'], length=14)
            df['AR'] = brar['AROOND_14']  # Approximate AR value using Aroon Down or adapt as needed

            if time_scale == 'days':
                # Convert the date column to a cyclical year 1-364 day representation
                df['Date'] = pd.to_datetime(df['Date'])
                df['Day'] = df['Date'].dt.dayofyear
                df['Date_sin'] = np.sin(2 * np.pi * df['Day'] / 365)
                df['Date_cos'] = np.cos(2 * np.pi * df['Day'] / 365)
                df['Date'] = (df['Date'] - timestamp_2000).dt.days
                df.drop(columns=['Day'], inplace=True)

            df = df[['Date', 'Date_sin', 'Date_cos', 'Open', 'High', 'Low', 'Close', 'EMA', 'Stoch_RSI', 'Stoch_RSI_D',  'SAR', 'SuperTrend','BOLL_High','BOLL_Low', 'MACD', 'MACD_Signal','WILLR','PSL','MASS','NATR','ROC','DMI_ADX','DMI_DI_P','DMI_DI_N','AR']] #
            df = df[14:,:]
            # df.dropna(inplace=True)
            # print(df.head())
            # exit(0)
            df.to_csv(f'./data/indicators/{time_scale}/{file_id}', index=False)
            processed_files[time_scale].add(file_id)
        except Exception as e:
            # print(traceback.format_exc())
            # exit(1)
            pass

        if i % 20 == 0:
            print(f"Processed {round(((i+1)/count) * 100, 2) }%", end='\r')

        # break
    # exit(0)

print('\nComparing files in time scales...')
# Remove files that are not present in all time scales
all_files = set.union(*[processed_files[scale] for scale in scales])
common_files = set.intersection(*[processed_files[scale] for scale in scales])

for id in all_files - common_files:
    for time_scale in scales:
        if os.path.exists(f'data/indicators/{time_scale}/{id}'):
            os.remove(f'data/indicators/{time_scale}/{id}')

print('Common files:', len(common_files))