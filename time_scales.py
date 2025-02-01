import pandas as pd
from pandas.tseries.offsets import CustomBusinessDay
from pandas.tseries.holiday import USFederalHolidayCalendar
from glob import glob 
import os
import numpy as np

print("Running time_scales.py...")

timestamp_2000 = pd.Timestamp('2000-01-01')

# csv example:
    # Date,Open,High,Low,Close
    # 2000-01-03,56.33047103881836,56.46459197998047,48.19384765625,51.50214767456055
    # 2000-01-04,48.73032760620117,49.26681137084961,46.316165924072266,47.56795501708984
    # 2000-01-05,47.38912582397461,47.56795501708984,43.14198684692383,44.6173095703125
csvs = glob('data/days/*.csv')
count = len(csvs)
for i, file in enumerate(csvs):
    # print(f"Reading {file}...")
    df = pd.read_csv(file)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    # Remove duplicate indices
    df = df[~df.index.duplicated(keep='first')]

    # Define a custom business day frequency that excludes weekends and US federal holidays
    us_bd = CustomBusinessDay(calendar=USFederalHolidayCalendar())
    # Alternatively, you can use the custom business day frequency directly if needed
    df = df.asfreq(us_bd)

    # Resample to weekly frequency, considering business days (Mon-Fri)
    weeks = df.resample('W-FRI').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last'
    }).dropna()

    # Reset the index if needed
    weeks.reset_index(inplace=True)
    
    # Convert the date column to a cyclical year 1-51 week representation
    weeks['Week'] = weeks['Date'].dt.isocalendar().week
    weeks['Date_sin'] = np.sin(2 * np.pi * weeks['Week'] / 52)
    weeks['Date'] = (weeks['Date'] - timestamp_2000).dt.days
    weeks.drop(columns=['Week'], inplace=True)
    # df.drop(columns=['Date'], inplace=True)
    
    weeks.to_csv(f'data/weeks/{os.path.basename(file)}', index=False)
    

    # Resample to monthly frequency, considering business days
    months = df.resample('ME').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last'
    }).dropna()
    # Reset the index if needed
    months.reset_index(inplace=True)
    
    # Convert the date column to a cyclical year 1-12 month representation
    months['Month'] = months['Date'].dt.month
    months['Date_sin'] = np.sin(2 * np.pi * months['Month'] / 12)
    months['Date'] = (months['Date'] - timestamp_2000).dt.days
    months.drop(columns=['Month'], inplace=True)
    # months.drop(columns=['Date'], inplace=True)
    
    # Export the monthly resampled data
    months.to_csv(f'data/months/{os.path.basename(file)}', index=False)
    
    # break
    if i % 10 == 0:
        print(f"Processed {round(((i+1)/count) * 100) }%")