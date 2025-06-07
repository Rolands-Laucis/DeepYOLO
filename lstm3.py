import glob
import os
import pandas as pd
import numpy as np
from pprint import pprint
import json
import argparse
import math

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Concatenate, Reshape, GRU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
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
parser.add_argument('--rnn_layers', type=int, help='rnn layer count that feed into each other in seq')
parser.add_argument('--epochs', type=int, default=10, help='epochs')
args = parser.parse_args()

# read cli args - if a json file path is passed, load the config from it
config = {}
if args.model:
    args.model = os.path.join('models', args.model)
    if os.path.exists(args.model):
        with open(args.model, 'r') as f:
            config = json.load(f)
            print('Loaded config from', args.model)
    else: raise ValueError(f'Provided [{args.model}] doesnt exist!')
if args.full:
    config['full_run'] = args.full
if args.csvs_per_run:
    config['csvs_per_run'] = args.csvs_per_run
if args.rnn:
    config['rnn_type'] = args.rnn
if args.units:
    config['rnn_units'] = args.units
if args.rnn_layers:
    config['rnn_layers'] = args.rnn_layers
config['epochs'] = args.epochs
config['save'] = args.save
if args.norm:
    config['norm'] = args.norm

# Configuration
lookback_days = config.get('lookback_days', 50)
prediction_days = config.get('prediction_days', 3)
full_run = config.get('full_run', False)
feature_count = config.get('feature_count', 23)
csvs_per_run = config.get('csvs_per_run', 5)
normalize = config.get('norm', False)
pred_cols = config.get('pred_cols', ['High', 'Low'])
# pprint(config)
# exit(0)

# CSV PATHS
csv_paths = glob.glob('./data/indicators/days/*.csv')
if args.csvs != -1: csv_paths = csv_paths[:args.csvs]
elif not full_run: csv_paths = csv_paths[:9]
csvs_total_len = len(csv_paths)
csvs_start = config.get('run_csvs_end', 0) #start where the last run ended
print('Total CSVS to be processed by all runs:', csvs_total_len)
# exit(0)

# constants
# https://stackoverflow.com/questions/29245848/what-are-all-the-dtypes-that-pandas-recognizes
df = pd.read_csv(csv_paths[0], engine='c', nrows=1, low_memory=False)
columns = df.columns
assert len(columns) == 1+feature_count, len(columns) #+1 for 'Date', which is not needed for the models features
dtypes = ['int16'] + [np.float32]*feature_count #date is int16, rest are float32, but pandas wont convert it for some reason
dtypes = dict(zip(columns, dtypes))
assert len(dtypes) == 1+feature_count, len(dtypes)
# print(df.head())
# print(columns)
# exit(0)
config['columns'] = columns[1:] # the date col will be removed
del df, columns

for run_csvs_start in range(csvs_start, csvs_total_len-1, csvs_per_run):
    # Prepare datasets
    X_daily, X_weekly, X_monthly, y = [], [], [], []

    run_csvs_end = min(run_csvs_start + csvs_per_run, csvs_total_len - 1)
    run_csv_paths = csv_paths[run_csvs_start:run_csvs_end]
    run_csv_count = len(run_csv_paths)
    print(f'Processing {run_csv_count} csvs from {run_csvs_start} to {run_csvs_end-1} with increment of {csvs_per_run} from total {csvs_total_len}...')
    # exit(0)

    print('Loading csv datasets...')
    for i, days_path in enumerate(run_csv_paths):
        # add progress print out
        if i % 100 == 0:
            print(f'{round((i/run_csv_count)*100)}% csvs {i}/{run_csv_count}', end='\r')

        daily_df = pd.read_csv(days_path, index_col=None, dtype=dtypes, engine='c', low_memory=True)#.fillna(0) # dtype=dtypes,
        
        max_day = len(daily_df)-prediction_days-lookback_days #offset from the end of the rows such that there is still enough day data points for both lookahead
        for first_day in range(lookback_days, max_day, lookback_days // lookback_days): #first day is actually the first row and we look forward to the current day + look ahead days for prediction
            current_day = first_day + lookback_days
            last_pred_day = current_day + prediction_days

            # Daily sequence
            days = daily_df.iloc[first_day:current_day].values
            days = days[:,1:] # remove the date column

            prediction_day_candles = daily_df.iloc[current_day+1:current_day+prediction_days+1][pred_cols].values
            
            X_daily.append(days)
            y.append(prediction_day_candles)

    X_daily = np.array(X_daily)
    y = np.array(y)
    shapes = {
        'X_daily': X_daily.shape,
        'y': y.shape
    }
    # pprint(shapes)
    del run_csv_count,  daily_df, days, prediction_day_candles

    # # Train-test split
    print('Splitting data into train and test sets...')
    test_size = config.get('test_size', 0.1)
    (X_daily_train, X_daily_test, 
    y_train, y_test) = train_test_split(X_daily, y, test_size=test_size)
    # y_train, y_test) = train_test_split(X_daily, X_weekly, y, test_size=test_size)

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
        'prediction_days': prediction_days,
        'full_run': full_run,
        'normalized': normalize,
        'feature_count': feature_count,
        'pred_cols': pred_cols,
        'loaded samples': y.shape[0],
        'train samples': len(y_train),
        'test samples': len(y_test),
        'test_size': test_size
    } | shapes
    pprint(config)

    # cleanup
    del X_daily, shapes, run_csv_paths, test_size

    # make and train model
    if config.get('model_path'):
        print('Loading model from last run...')
        model = load_model(config['model_path'])
        print('Model loaded')
    else:
        print('Building model...')

        # Model architecture
        daily_input = Input(shape=(lookback_days, feature_count)) #the input the LSTM will receive a squence of 21 days, each a vector of 13 features

        rnn_type = config.get('rnn_type', 'lstm')
        rnn_layers = config.get('rnn_layers', 3)
        rnn_units = config.get('rnn_units', 64) if full_run else 16

        # Helper function to build a branch with a given number of recurrent layers.
        # Build RNN layers with decreasing units
        def build_rnn_branch(input_tensor, num_layers, base_units, rnn_type='lstm'):
            x = input_tensor
            current_units = base_units
            for i in range(num_layers):
                return_sequences = (i < num_layers - 1)
                layer_units = max(1, math.floor(current_units))  # Ensure at least 1 unit
                if rnn_type.lower() == 'lstm':
                    x = LSTM(layer_units, return_sequences=return_sequences)(x)
                elif rnn_type.lower() == 'gru':
                    x = GRU(layer_units, return_sequences=return_sequences)(x)
                else:
                    raise ValueError('rnn_type must be either "lstm" or "gru"')
                current_units /= 1.5  # Reduce units by 1.5x for the next layer
            return x, math.floor(current_units * 1.5)  # Return last used layer's size

        # Build the RNN branch
        layer, last_lstm_units = build_rnn_branch(daily_input, rnn_layers, rnn_units, rnn_type)

        # Add a smaller Dense layer after LSTM
        if full_run:
            # layer = Dense(rnn_units, activation='relu')(layer)
            dense_units = max(1, last_lstm_units // 2)
            layer = Dense(dense_units, activation='leaky_relu')(layer)
        if rnn_units > 64:
            layer = Dense(64, activation='leaky_relu')(layer)
        layer = Dense(prediction_days * len(pred_cols))(layer)
        layer = Reshape((prediction_days, len(pred_cols)))(layer)

        # https://keras.io/api/metrics/classification_metrics/#f1score-class
        model = Model(inputs=[daily_input], outputs=layer) #
        model.compile(optimizer=Adam(0.001), loss='mse', metrics=['accuracy']) #, 'f1_score'
        
        # cleanup
        del daily_input, layer

        print('Model built.')

    # model.summary()

    # Train model
    print('Starting train...')
    epochs = config.get('epochs', 20) if full_run else 1
    batch_size = 32
    model.fit(
        [X_daily_train],
        # [X_daily_train, X_weekly_train],
        y_train,
        validation_data=([X_daily_test], y_test),
        # validation_data=([X_daily_test, X_weekly_test], y_test),
        epochs=epochs,
        batch_size=batch_size
    )

    # Evaluate model
    metrics = model.evaluate([X_daily_test], y_test)
    # metrics = model.evaluate([X_daily_test, X_weekly_test], y_test)
    print(f'Test metrics: {metrics}')

    # Predict and calculate additional metrics
    y_pred = model.predict([X_daily_test])
    # y_pred = model.predict([X_daily_test, X_weekly_test])

    # Reshape predictions and true values to 2D arrays for metric calculations
    y_pred_flat = y_pred.reshape(-1, len(pred_cols))
    y_test_flat = y_test.reshape(-1, len(pred_cols))

    # Calculate regression metrics
    mae = mean_absolute_error(y_test_flat, y_pred_flat)
    rmse = np.sqrt(mean_squared_error(y_test_flat, y_pred_flat))

    print(f'Mean Absolute Error: {mae}')
    print(f'Root Mean Squared Error: {rmse}')

    del X_daily_train, X_daily_test, y_train, y_test

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