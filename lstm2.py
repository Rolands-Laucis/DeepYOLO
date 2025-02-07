import glob
import os
import pandas as pd
import numpy as np
import bisect
from pprint import pprint
import json
import argparse

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Concatenate, Reshape, GRU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_absolute_error, mean_squared_error

def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in ('false', '0', 'no', 'off'):
        return False
    return True  # Any other value (including being supplied without an argument) is treated as True

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Process some args.')
parser.add_argument('--model', type=str, default='', help='Path to previous model to resume training.')
parser.add_argument('--full', type=bool, default=False, help='full mode')
parser.add_argument('--save', nargs='?', const=True, default=True, type=str_to_bool, help='Save model (default: True, explicitly use "false" to disable)')
parser.add_argument('--norm', action='store_true', help='normalize candles')
parser.add_argument('--csvs', type=int, default=-1, help='how many csvs to process')
parser.add_argument('--csvs_per_run', type=int, help='how many csvs to process per run')
parser.add_argument('--rnn', type=str, choices=['lstm', 'gru'], help='type of the rnn layer')
parser.add_argument('--units', type=int, help='units in the rnn layer')
parser.add_argument('--epochs', type=int, default=10, help='epochs')
args = parser.parse_args()

# read cli args - if a json file path is passed, load the config from it
config = {}
if args.model:
    with open(args.model, 'r') as f:
        config = json.load(f)
        print('Loaded config from', args.model)
if args.full:
    config['full_run'] = args.full
if args.csvs_per_run:
    config['csvs_per_run'] = args.csvs_per_run
if args.rnn:
    config['rnn_type'] = args.rnn
if args.units:
    config['rnn_units'] = args.units
config['epochs'] = args.epochs
config['save'] = args.save
if args.norm:
    config['norm'] = args.norm

# Configuration
lookback_days = config.get('lookback_days', 21)
lookback_weeks = config.get('lookback_weeks', 8)
# lookback_months = config.get('lookback_months', 6)
prediction_days = config.get('prediction_days', 5)
full_run = config.get('full_run', False)
feature_count = config.get('feature_count', 13)
# feature_count = config.get('feature_count', 12)
csvs_per_run = config.get('csvs_per_run', 5)
normalize = config.get('norm', False)
# pprint(config)
# exit(0)

# constnats
# https://stackoverflow.com/questions/29245848/what-are-all-the-dtypes-that-pandas-recognizes
columns = 'Date,Date_sin,Open,High,Low,Close,28_EMA,Stoch_RSI,Stoch_RSI_D,MACD,MACD_Signal,MACD_Hist,SAR,SuperTrend'.split(',')
assert len(columns) == 1+feature_count, len(columns) #+1 for 'Date', which is not needed for the models features
dtypes = ['int16'] + [np.float32]*feature_count #date is int16, rest are float32, but pandas wont convert it for some reason
dtypes = dict(zip(columns, dtypes))
assert len(dtypes) == 1+feature_count, len(dtypes)
col_order = ['Date_sin', 'Open', 'High', 'Low', 'Close', '28_EMA', 'Stoch_RSI', 'Stoch_RSI_D', 'MACD', 'MACD_Signal', 'MACD_Hist', 'SAR', 'SuperTrend']

csv_paths = glob.glob('./data/indicators/days/*.csv')
if args.csvs != 1: csv_paths = csv_paths[:args.csvs]
if not full_run: csv_paths = csv_paths[:10]
csvs_total_len = len(csv_paths)
csvs_start = config.get('run_csvs_end', 0) #start where the last run ended
print('Total CSVS to be processed by all runs:', csvs_total_len)
for run_csvs_start in range(csvs_start, csvs_total_len-1, csvs_per_run):
    # Prepare datasets
    X_daily, X_weekly, X_monthly, y = [], [], [], []

    run_csvs_end = min(run_csvs_start + csvs_per_run, csvs_total_len - 1)
    run_csv_paths = csv_paths[run_csvs_start:run_csvs_end]
    run_csv_count = len(run_csv_paths)
    print(f'Processing {run_csv_count} csvs from {run_csvs_start} to {run_csvs_end-1} with increment of {csvs_per_run}...')
    # exit(0)

    print('Loading csv datasets...')
    for i, days_path in enumerate(run_csv_paths):
        # add progress print out
        if i % 100 == 0:
            print(f'{round((i/run_csv_count)*100)}% csvs {i}/{run_csv_count}', end='\r')

        ticker_prefix = os.path.basename(days_path).split('.csv')[0]
        weeks_path = f'./data/indicators/weeks/{ticker_prefix}.csv'
        # months_path = f'./data/indicators/months/{ticker_prefix}.csv'
        
        daily_df = pd.read_csv(days_path, index_col=None, dtype=dtypes, engine='c', low_memory=True).fillna(0)
        weekly_df = pd.read_csv(weeks_path, index_col=None, dtype=dtypes, engine='c', low_memory=True).fillna(0)
        # monthly_df = pd.read_csv(months_path, index_col=None, dtype=dtypes, engine='c', low_memory=True).fillna(0)
        
        # get a table of week dates and their row indexes
        weekly_dates = weekly_df['Date'].values.astype(int)
        # monthly_dates = monthly_df['Date'].values.astype(int)

        max_day = len(daily_df)-prediction_days-lookback_days #offset from the end of the rows such that there is still enough day data points for both lookahead
        for first_day in range(lookback_days, max_day, lookback_days // 8): #first day is actually the first row and we look forward to the current day + look ahead days for prediction
            current_day = first_day + lookback_days
            last_pred_day = current_day + prediction_days

            # Daily sequence
            days = daily_df.iloc[first_day:current_day].values
            # if any of the indicators isnt ready, then skip
            if np.any(days == 0) or np.any(np.isnan(days)):
                continue
            days = days[:,1:] # remove the date column
            # normalize the cols
            target_cols = [0, 1, 2, 3]  # Indices of 'Open', 'Close', 'High', 'Low'
            if normalize:
                max_price = np.max(days[:, target_cols], axis=0) / 100
                days[:, target_cols] /= max_price
            # print(days[:5])
            # exit(0)

            # Weekly data
            # in weekly_dates find the highest date number that is closest to current_day number
            week_idx = bisect.bisect_right(weekly_dates, current_day)
            # from weekly_df select the rows from week_idx-lookback_weeks+1 to week_idx
            weeks = weekly_df.iloc[max(0, week_idx-lookback_weeks):week_idx, 1:].values # remove the date column
            # if any of the indicators isnt ready, then skip
            if np.any(weeks == 0) or np.any(np.isnan(weeks)):
                continue
            # if weeks.shape[0] < lookback_weeks then left pad with int16 zeros
            if weeks.shape[0] < lookback_weeks:
                pad = np.zeros((lookback_weeks-weeks.shape[0], feature_count), dtype=np.float32)
                weeks = np.vstack((pad, weeks))
            # normalize the cols
            if normalize:
                weeks[:, target_cols] /= max_price
            # print(weeks[:5])
            # exit(0)
            # print('weeks', week_idx, weeks.shape, weeks.dtype)
            # exit(0)

            # Monthly data
            # in monthly_dates find the highest date number that is closest to current_day number
            # month_idx = bisect.bisect_right(monthly_dates, current_day)
            # # from monthly_df select the rows from month_idx-lookback_months+1 to month_idx
            # months = monthly_df.iloc[max(0, month_idx-lookback_months):month_idx, 1:].values
            # # if any of the indicators isnt ready, then skip
            # # if np.any(months == 0) or np.any(np.isnan(months)):
            # #     continue
            # # if months.shape[0] < lookback_months then left pad with int16 zeros
            # if months.shape[0] < lookback_months:
            #     pad = np.zeros((lookback_months-months.shape[0], feature_count), dtype=np.int16)
            #     months = np.vstack((pad, months))
            # print('months', month_idx, months.shape)

            prediction_day_candles = daily_df.iloc[current_day+1:current_day+prediction_days+1][['Open', 'Close', 'High', 'Low']].values
            if normalize:
                prediction_day_candles[:, target_cols] /= max_price
            
            X_daily.append(days)
            X_weekly.append(weeks)
            # X_monthly.append(months)
            y.append(prediction_day_candles)

            # add progress print out
            # if first_day % 2000 == 0:
            #     print(f'{first_day}/{max_day} rows {round((first_day/max_day)*100)}% ', end='\r')

    X_daily = np.array(X_daily)
    X_weekly = np.array(X_weekly)
    # X_monthly = np.array(X_monthly)
    y = np.array(y)
    shapes = {
        'X_daily': X_daily.shape,
        'X_weekly': X_weekly.shape,
        # 'X_monthly': X_monthly.shape,
        'y': y.shape
    }
    # pprint(shapes)
    del run_csv_count, weekly_dates, daily_df, weekly_df, days, weeks, prediction_day_candles
    # del monthly_dates, monthly_df, months

    # # Train-test split
    print('Splitting data into train and test sets...')
    test_size = config.get('test_size', 0.1)
    from sklearn.model_selection import train_test_split
    (X_daily_train, X_daily_test, 
    X_weekly_train, X_weekly_test, 
    # X_monthly_train, X_monthly_test, 
    # y_train, y_test) = train_test_split(X_daily, X_weekly, X_monthly, y, test_size=test_size)
    y_train, y_test) = train_test_split(X_daily, X_weekly, y, test_size=test_size)

    # print config
    config = config | {
        # 'daily dtype':str(X_daily.dtype), 
        # 'weekly dtype':str(X_weekly.dtype), 
        # 'monthly dtype':str(X_monthly.dtype), 
        # 'y dtype':str(y.dtype),
        'run_csvs_start': run_csvs_start,
        'run_csvs_end': run_csvs_end,
        'total_csvs': csvs_total_len,
        'run_csvs': len(run_csv_paths),
        'lookback_days': lookback_days,
        'lookback_weeks': lookback_weeks,
        # 'lookback_months': lookback_months,
        'prediction_days': prediction_days,
        'full_run': full_run,
        'normalized': normalize,
        'feature_count': feature_count,
        'loaded samples': y.shape[0],
        'train samples': len(y_train),
        'test samples': len(y_test),
        'test_size': test_size
    } | shapes
    pprint(config)

    # cleanup
    del X_daily, X_weekly, y, shapes, run_csv_paths, test_size
    del X_monthly

    # make and train model
    if config.get('model_path'):
        print('Loading model from last run...')
        model = load_model(config['model_path'])
        print('Model loaded')
    else:
        print('Building model...')

        # Model architecture
        daily_input = Input(shape=(lookback_days, feature_count)) #the input the LSTM will receive a squence of 21 days, each a vector of 13 features
        weekly_input = Input(shape=(lookback_weeks, feature_count))
        # monthly_input = Input(shape=(lookback_months, feature_count))

        rnn_type = config.get('rnn_type', 'lstm')
        rnn_units = config.get('rnn_units', 128) if full_run else 32

        if rnn_type == 'gru':
            daily = GRU(rnn_units)(daily_input)
            weekly = GRU(rnn_units)(weekly_input)
            # monthly = GRU(rnn_units)(monthly_input)
        elif rnn_type == 'lstm':
            daily = LSTM(rnn_units, return_sequences=False)(daily_input) #in a loop of 21 times (bcs of day sequence length) the LSTM will output a single vector of 128 features to pass back to itself with the next day
            weekly = LSTM(rnn_units, return_sequences=False)(weekly_input)
            # monthly = LSTM(rnn_units, return_sequences=False)(monthly_input)
        else:
            raise ValueError('rnn_type must be GRU or LSTM')

        layer = Concatenate()([daily, weekly]) #, monthly
        if full_run:
            layer = Dense(rnn_units, activation='relu')(layer)
        #     layer = Dense(rnn_units // 2, activation='relu')(layer)
        # if rnn_units > 64:
        layer = Dense(32, activation='relu')(layer)
        layer = Dense(prediction_days * 4)(layer)
        layer = Reshape((prediction_days, 4))(layer)

        # https://keras.io/api/metrics/classification_metrics/#f1score-class
        model = Model(inputs=[daily_input, weekly_input], outputs=layer) #, monthly_input
        model.compile(optimizer=Adam(0.001), loss='mse', metrics=['accuracy']) #, 'f1_score'
        
        # cleanup
        del daily_input, weekly_input, daily, weekly, layer
        # del monthly_input, monthly

        print('Model built.')

    # model.summary()

    # Train model
    print('Starting train...')
    epochs = config.get('epochs', 20) if full_run else 1
    batch_size = 32
    model.fit(
        # [X_daily_train, X_weekly_train, X_monthly_train],
        [X_daily_train, X_weekly_train],
        y_train,
        # validation_data=([X_daily_test, X_weekly_test, X_monthly_test], y_test),
        validation_data=([X_daily_test, X_weekly_test], y_test),
        epochs=epochs,
        batch_size=batch_size
    )

    # Evaluate model
    # metrics = model.evaluate([X_daily_test, X_weekly_test, X_monthly_test], y_test)
    metrics = model.evaluate([X_daily_test, X_weekly_test], y_test)
    print(f'Test metrics: {metrics}')

    # Predict and calculate additional metrics
    # y_pred = model.predict([X_daily_test, X_weekly_test, X_monthly_test])
    y_pred = model.predict([X_daily_test, X_weekly_test])

    # Reshape predictions and true values to 2D arrays for metric calculations
    y_pred_flat = y_pred.reshape(-1, 4)
    y_test_flat = y_test.reshape(-1, 4)

    # Calculate regression metrics
    mae = mean_absolute_error(y_test_flat, y_pred_flat)
    rmse = np.sqrt(mean_squared_error(y_test_flat, y_pred_flat))

    print(f'Mean Absolute Error: {mae}')
    print(f'Root Mean Squared Error: {rmse}')

    del X_daily_train, X_daily_test, X_weekly_train, X_weekly_test, y_train, y_test
    # del X_monthly_train, X_monthly_test

    # Save the config and model summary to a JSON file
    if config.get('save', True):
        print('Saving model...')
        config['metrics'] = {
            'loss': metrics[0],
            'accuracy': metrics[1],
            'mae': mae,
            'rmse': rmse
        }
        config['epochs'] = epochs

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
        rnn_type = 'lstm' if 'lstm' in layer_info else 'gru'
        rnn_units = layer_info['lstm']['units'] if rnn_type == 'lstm' else layer_info['gru']['units']

        path = f'./models/{rnn_type}_loss{round(config["metrics"]["loss"], 3)}_a{round(config["metrics"]["accuracy"], 3)}_rmse{round(rmse, 3)}_u{rnn_units}_e{epochs}_csvs{csvs_total_len}_d{lookback_days}_w{lookback_weeks}_c{prediction_days}' + ('_norm' if normalize else '') #_m{lookback_months}
        model_path = f'{path}.keras'
        config['model_path'] = model_path
        with open(f'{path}.json', 'w') as f:
            json.dump(config, f, indent=2)

        # Save the model
        model.save(model_path)

        del model, rnn_type, rnn_units, layer_info, model_data
        del mae, rmse, metrics, y_pred, y_pred_flat, y_test_flat
        del epochs, batch_shape, batch_size

input('Press any key to exit...')