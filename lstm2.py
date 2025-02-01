import glob
import os
import pandas as pd
import numpy as np
import bisect
from pprint import pprint
import sys
import json

# read cli args - if a json file path is passed, load the config from it
config = {}
if len(sys.argv) > 1:
    config_path = sys.argv[1]
    with open(config_path, 'r') as f:
        config = json.load(f)
        print('Loaded config from', config_path)

# Configuration
lookback_days = config.get('lookback_days', 21)
lookback_weeks = config.get('lookback_weeks', 8)
lookback_months = config.get('lookback_months', 6)
prediction_days = config.get('prediction_days', 3)
full_run = config.get('full_run', True)
feature_count = config.get('feature_count', 13)

# constnats
# https://stackoverflow.com/questions/29245848/what-are-all-the-dtypes-that-pandas-recognizes
columns = 'Date,Date_sin,Open,High,Low,Close,50_EMA,Stoch_RSI,Stoch_RSI_D,MACD,MACD_Signal,MACD_Hist,SAR,SuperTrend'.split(',')
assert len(columns) == 1+feature_count, len(columns) #+1 for 'Date', which is not needed for the models features
dtypes = ['int16'] + [np.float32]*feature_count #date is int16, rest are float32, but pandas wont convert it for some reason
dtypes = dict(zip(columns, dtypes))
assert len(dtypes) == 1+feature_count, len(dtypes)

# Prepare datasets
X_daily, X_weekly, X_monthly, y = [], [], [], []
csv_paths = glob.glob('./data/indicators/days/*.csv')
csvs_total_len = len(csv_paths)
csvs_start = config.get('csvs_end', 0) #start where the last run ended
csvs_end = min(csvs_start + 35, csvs_total_len - 1)
if not full_run: csv_paths = csv_paths[:2]
else: csv_paths[csvs_start:csvs_end]
csv_count = len(csv_paths)
print(f'Processing {csv_count} csvs from {csvs_start} to {csvs_end}...')
# exit(0)

print('Loading csv datasets...')
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
del csv_count, dtypes, weekly_dates, monthly_dates, daily_df, weekly_df, monthly_df, days, weeks, months, prediction_day_candles, pad

# # Train-test split
print('Splitting data into train and test sets...')
test_size = config.get('test_size', 0.1)
from sklearn.model_selection import train_test_split
(X_daily_train, X_daily_test, 
 X_weekly_train, X_weekly_test, 
 X_monthly_train, X_monthly_test, 
 y_train, y_test) = train_test_split(X_daily, X_weekly, X_monthly, y, test_size=test_size)

# print config
config = {
    # 'daily dtype':str(X_daily.dtype), 
    # 'weekly dtype':str(X_weekly.dtype), 
    # 'monthly dtype':str(X_monthly.dtype), 
    # 'y dtype':str(y.dtype),
    'csvs_start': csvs_start,
    'csvs_end': csvs_end,
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
    'test_size': test_size
} | shapes
pprint(config)

# cleanup
del X_daily, X_weekly, X_monthly, y, shapes, csv_paths, test_size

# make and train model
if config.get('model_path'):
    print('Loading model from last run...')
    from tensorflow.keras.models import load_model
    model = load_model(config['model_path'])
    print('Model loaded')
else:
    print('Building model...')
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, LSTM, Dense, Concatenate, Reshape, GRU
    from tensorflow.keras.optimizers import Adam
    from sklearn.metrics import mean_absolute_error, mean_squared_error

    # Model architecture
    daily_input = Input(shape=(lookback_days, feature_count)) #the input the LSTM will receive a squence of 21 days, each a vector of 13 features
    weekly_input = Input(shape=(lookback_weeks, feature_count))
    monthly_input = Input(shape=(lookback_months, feature_count))

    rnn_type = 'LSTM' # 'GRU' or 'LSTM'
    rnn_units = 64 if full_run else 32

    if rnn_type == 'GRU':
        daily = GRU(rnn_units)(daily_input)
        weekly = GRU(rnn_units)(weekly_input)
        monthly = GRU(rnn_units)(monthly_input)
    elif rnn_type == 'LSTM':
        daily = LSTM(rnn_units, return_sequences=False)(daily_input) #in a loop of 21 times (bcs of day sequence length) the LSTM will output a single vector of 128 features to pass back to itself with the next day
        weekly = LSTM(rnn_units, return_sequences=False)(weekly_input)
        monthly = LSTM(rnn_units, return_sequences=False)(monthly_input)
    else:
        raise ValueError('rnn_type must be GRU or LSTM')

    layer = Concatenate()([daily, weekly, monthly])
    if full_run:
        layer = Dense(rnn_units, activation='relu')(layer)
    layer = Dense(32, activation='relu')(layer)
    layer = Dense(prediction_days * 4)(layer)
    layer = Reshape((prediction_days, 4))(layer)

    # https://keras.io/api/metrics/classification_metrics/#f1score-class
    model = Model(inputs=[daily_input, weekly_input, monthly_input], outputs=layer)
    model.compile(optimizer=Adam(0.001), loss='mse', metrics=['accuracy']) #, 'f1_score'
    
    # cleanup
    del daily_input, weekly_input, monthly_input, daily, weekly, monthly, layer

model.summary()

# Train model
epochs = 50 if full_run else 5
batch_size = 32
model.fit(
    [X_daily_train, X_weekly_train, X_monthly_train],
    y_train,
    validation_data=([X_daily_test, X_weekly_test, X_monthly_test], y_test),
    epochs=epochs,
    batch_size=batch_size
)

# Evaluate model
metrics = model.evaluate([X_daily_test, X_weekly_test, X_monthly_test], y_test)
print(f'Test metrics: {metrics}')

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


# Save the config and model summary to a JSON file
print('Saving model...')
config['metrics'] = {
    'loss': metrics[0],
    'accuracy': metrics[1],
    'mae': mae,
    'rmse': rmse
}

# Extract relevant model layer info
model_data = json.loads(model.to_json())
layer_info = {}
for layer in model_data["config"]["layers"]:
    layer_type = layer["class_name"]
    layer_name = layer["name"]
    # Determine units if applicable
    units = layer["config"].get("units")
    if units is None:
        # Try to extract shape for InputLayer
        batch_shape = layer["config"].get("batch_shape")
        units = batch_shape[1] if batch_shape else None  # Take second dimension if exists
    layer_info[layer_name] = {"type": layer_type, "units": units}

config['model_summary'] = layer_info

path = f'./models/{rnn_type}_loss{round(config["metrics"]["loss"], 3)}_a{round(config["metrics"]["accuracy"], 3)}_rmse{round(rmse, 3)}_u{rnn_units}_e{epochs}_d{lookback_days}_w{lookback_weeks}_m{lookback_months}_c{prediction_days}'
model_path = f'{path}.keras'
config['model_path'] = model_path
with open(f'{path}.json', 'w') as f:
    json.dump(config, f, indent=2)

# Save the model
model.save(model_path)