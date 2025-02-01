import glob
import os
import pandas as pd
import numpy as np
import bisect
from pprint import pprint

# Configuration
lookback_days = 21
lookback_weeks = 8
lookback_months = 6
prediction_days = 3
full_run = True
feature_count = 13

# constnats
# https://stackoverflow.com/questions/29245848/what-are-all-the-dtypes-that-pandas-recognizes
columns = 'Date,Date_sin,Open,High,Low,Close,50_EMA,Stoch_RSI,Stoch_RSI_D,MACD,MACD_Signal,MACD_Hist,SAR,SuperTrend'.split(',')
assert len(columns) == 1+feature_count, len(columns) #+1 for 'Date', which is not needed for the models features
dtypes = ['int16'] + [np.float32]*feature_count #date is int16, rest are float32, but pandas wont convert it for some reason
dtypes = dict(zip(columns, dtypes))
assert len(dtypes) == 1+feature_count, len(dtypes)

# Prepare datasets
print('Loading csv datasets...')
X_daily, X_weekly, X_monthly, y = [], [], [], []
csv_paths = glob.glob('./data/indicators/days/*.csv')[:35]
if not full_run: csv_paths = csv_paths[:5]
csv_count = len(csv_paths)
for i, days_path in enumerate(csv_paths):
    # add progress print out
    print(f'\n{round((i/csv_count)*100)}% csvs {i}/{csv_count}\n', end='\r')

    ticker_prefix = os.path.basename(days_path).split('.csv')[0]
    weeks_path = f'./data/indicators/weeks/{ticker_prefix}.csv'
    months_path = f'./data/indicators/months/{ticker_prefix}.csv'
    # print(ticker_prefix, days_path, weeks_path, months_path)
    
    daily_df = pd.read_csv(days_path, index_col=None, dtype=dtypes, engine='c', low_memory=True).fillna(0)#.astype(dtypes)#.sort_values('Date')
    weekly_df = pd.read_csv(weeks_path, index_col=None, dtype=dtypes, engine='c', low_memory=True).fillna(0)#.astype(dtypes)#.sort_values('Date')
    monthly_df = pd.read_csv(months_path, index_col=None, dtype=dtypes, engine='c', low_memory=True).fillna(0)#.astype(dtypes)#.sort_values('Date')
    
    # get a table of week dates and their row indexes
    weekly_dates = weekly_df['Date'].values.astype(int)
    monthly_dates = monthly_df['Date'].values.astype(int)

    max_day = len(daily_df)-prediction_days-lookback_days #offset from the end of the rows such that there is still enough day data points for both lookahead
    for first_day in range(0, max_day): #first day is actually the first row and we look forward to the current day + look ahead days for prediction
        current_day = first_day + lookback_days
        last_pred_day = current_day + prediction_days
        # print(first_day, current_day, last_pred_day)

        # Daily sequence
        days = daily_df.iloc[first_day:current_day].values
        days = days[:,1:] # remove the date column
        # print('days', days.shape, days.dtype)
        X_daily.append(days)

        # Weekly data
        # in weekly_dates find the highest date number that is closest to current_day number
        week_idx = bisect.bisect_right(weekly_dates, current_day)
        # from weekly_df select the rows from week_idx-lookback_weeks+1 to week_idx
        weeks = weekly_df.iloc[max(0, week_idx-lookback_weeks):week_idx, 1:].values
        # if weeks.shape[0] < lookback_weeks then left pad with int16 zeros
        if weeks.shape[0] < lookback_weeks:
            pad = np.zeros((lookback_weeks-weeks.shape[0], feature_count), dtype=np.float32)
            weeks = np.vstack((pad, weeks))
        # print('weeks', week_idx, weeks.shape, weeks.dtype)
        X_weekly.append(weeks)
        # exit(0)

        # Monthly data
        # in monthly_dates find the highest date number that is closest to current_day number
        month_idx = bisect.bisect_right(monthly_dates, current_day)
        # from monthly_df select the rows from month_idx-lookback_months+1 to month_idx
        months = monthly_df.iloc[max(0, month_idx-lookback_months):month_idx, 1:].values
        # if months.shape[0] < lookback_months then left pad with int16 zeros
        if months.shape[0] < lookback_months:
            pad = np.zeros((lookback_months-months.shape[0], feature_count), dtype=np.int16)
            months = np.vstack((pad, months))
        # print('months', month_idx, months.shape)
        X_monthly.append(months)

        prediction_day_candles = daily_df.iloc[current_day+1:current_day+prediction_days+1][['Open', 'Close', 'High', 'Low']].values
        y.append(prediction_day_candles)

        # add progress print out
        if first_day % 2000 == 0:
            print(f'{first_day}/{max_day} rows {round((first_day/max_day)*100)}% ', end='\r')

X_daily = np.array(X_daily)
X_weekly = np.array(X_weekly)
X_monthly = np.array(X_monthly)
y = np.array(y)
shapes = {
    'X_daily': X_daily.shape,
    'X_weekly': X_weekly.shape,
    'X_monthly': X_monthly.shape,
    'y': y.shape
}
# pprint(shapes)
del csv_count, csv_paths, dtypes, weekly_dates, monthly_dates, daily_df, weekly_df, monthly_df, days, weeks, months, prediction_day_candles, pad

# # Train-test split
from sklearn.model_selection import train_test_split
(X_daily_train, X_daily_test, 
 X_weekly_train, X_weekly_test, 
 X_monthly_train, X_monthly_test, 
 y_train, y_test) = train_test_split(X_daily, X_weekly, X_monthly, y, test_size=0.1)

# print config
config = {
    'daily dtype':X_daily.dtype, 
    'weekly dtype':X_weekly.dtype, 
    'monthly dtype':X_monthly.dtype, 
    'y dtype':y.dtype,
    'lookback_days': lookback_days,
    'lookback_weeks': lookback_weeks,
    'lookback_months': lookback_months,
    'prediction_days': prediction_days,
    'full_run': full_run,
    'feature_count': feature_count,
    'loaded samples': y.shape[0],
    'csvs': len(csv_paths),
    'train samples': len(y_train),
    'test samples': len(y_test),
} | shapes
pprint(config)

# cleanup
del X_daily, X_weekly, X_monthly, y, shapes

# make and train model

# os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Concatenate, Reshape, GRU
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Model architecture
daily_input = Input(shape=(lookback_days, feature_count)) #the input the LSTM will receive a squence of 21 days, each a vector of 13 features
weekly_input = Input(shape=(lookback_weeks, feature_count))
monthly_input = Input(shape=(lookback_months, feature_count))

rnn_units = 64 if full_run else 32
# daily = GRU(rnn_units)(daily_input)
# weekly = GRU(rnn_units)(weekly_input)
# monthly = GRU(rnn_units)(monthly_input)

daily = LSTM(rnn_units, return_sequences=False)(daily_input) #in a loop of 21 times (bcs of day sequence length) the LSTM will output a single vector of 128 features to pass back to itself with the next day
weekly = LSTM(rnn_units, return_sequences=False)(weekly_input)
monthly = LSTM(rnn_units, return_sequences=False)(monthly_input)

layer = Concatenate()([daily, weekly, monthly])
if full_run:
    layer = Dense(rnn_units, activation='relu')(layer)
layer = Dense(32, activation='relu')(layer)
layer = Dense(prediction_days * 4)(layer)
layer = Reshape((prediction_days, 4))(layer)

model = Model(inputs=[daily_input, weekly_input, monthly_input], outputs=layer)
model.compile(optimizer=Adam(0.001), loss='mse')
model.summary()

# cleanup
del daily_input, weekly_input, monthly_input, daily, weekly, monthly, layer

# Train model
model.fit(
    [X_daily_train, X_weekly_train, X_monthly_train],
    y_train,
    validation_data=([X_daily_test, X_weekly_test, X_monthly_test], y_test),
    epochs=50 if full_run else 10,
    batch_size=32
)

# Evaluate model
loss = model.evaluate([X_daily_test, X_weekly_test, X_monthly_test], y_test)
print(f'Test Loss: {loss}')

# Predict and calculate additional metrics
y_pred = model.predict([X_daily_test, X_weekly_test, X_monthly_test])

# Reshape predictions and true values to 2D arrays for metric calculations
y_pred_flat = y_pred.reshape(-1, 4)
y_test_flat = y_test.reshape(-1, 4)

# Calculate regression metrics
mae = mean_absolute_error(y_test_flat, y_pred_flat)
rmse = np.sqrt(mean_squared_error(y_test_flat, y_pred_flat))

print(f'Mean Absolute Error: {mae}')
print(f'Root Mean Squared Error: {rmse}')