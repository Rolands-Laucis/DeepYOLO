import glob
import os
import pandas as pd
import numpy as np
import bisect
# os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Concatenate, Reshape
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Configuration
lookback_days = 21
lookback_weeks = 8
lookback_months = 6
prediction_days = 5
full_run = False

def load_and_process_ticker(days_path):
    ticker_prefix = os.path.basename(days_path).split('.')[0]
    weeks_path = f'./data/indicators/weeks/{ticker_prefix}.csv'
    months_path = f'./data/indicators/months/{ticker_prefix}.csv'
    
    if not (os.path.exists(weeks_path) and os.path.exists(months_path)):
        return None
    
    daily_df = pd.read_csv(days_path, index_col=None).sort_values('Date')
    weekly_df = pd.read_csv(weeks_path, index_col=None).sort_values('Date')
    monthly_df = pd.read_csv(months_path, index_col=None).sort_values('Date')
    
    daily_dates = daily_df['Date'].values.astype(int)
    weekly_dates = weekly_df['Date'].values.astype(int)
    monthly_dates = monthly_df['Date'].values.astype(int)
    
    samples = []
    
    for i in range(lookback_days-1, len(daily_df)-prediction_days):
        current_day = daily_dates[i]
        
        # Daily sequence
        daily_start = i - lookback_days + 1
        daily_seq = daily_df.iloc[daily_start:i+1]
        
        # Weekly alignment
        target_week = current_day // 7
        week_idx = bisect.bisect_right(weekly_dates, target_week) - 1
        if week_idx < lookback_weeks - 1 or week_idx >= len(weekly_df):
            continue
        weekly_seq = weekly_df.iloc[week_idx - lookback_weeks + 1 : week_idx + 1]
        
        # Monthly alignment
        target_month = current_day // 30
        month_idx = bisect.bisect_right(monthly_dates, target_month) - 1
        if month_idx < lookback_months - 1 or month_idx >= len(monthly_df):
            continue
        monthly_seq = monthly_df.iloc[month_idx - lookback_months + 1 : month_idx + 1]
        
        # Validate lengths
        if (len(daily_seq) != lookback_days or 
            len(weekly_seq) != lookback_weeks or 
            len(monthly_seq) != lookback_months):
            continue
        
        # Target values
        target = daily_df.iloc[i+1:i+1+prediction_days][['Open', 'Close', 'High', 'Low']].values
        
        # Feature extraction
        daily_features = daily_seq[['Open', 'High', 'Low', 'Close', '50_EMA', 
            'Stoch_RSI', 'Stoch_RSI_D', 'MACD', 'MACD_Signal', 'MACD_Hist', 
            'SAR', 'SuperTrend', 'Day_sin']].values
        
        weekly_features = weekly_seq[['Open', 'High', 'Low', 'Close', 'Week_sin',
            '50_EMA', 'Stoch_RSI', 'Stoch_RSI_D', 'MACD', 'MACD_Signal', 
            'MACD_Hist', 'SAR', 'SuperTrend']].values
        
        monthly_features = monthly_seq[['Open', 'High', 'Low', 'Close', 'Month_sin',
            '50_EMA', 'Stoch_RSI', 'Stoch_RSI_D', 'MACD', 'MACD_Signal', 
            'MACD_Hist', 'SAR', 'SuperTrend']].values
        
        samples.append((daily_features, weekly_features, monthly_features, target))
    
    return samples

# Collect data from all tickers
all_samples = []
csv_paths = glob.glob('./data/indicators/days/*.csv')
if not full_run: csv_paths = csv_paths[:50]
for days_path in csv_paths:
    ticker_samples = load_and_process_ticker(days_path)
    if ticker_samples:
        all_samples.extend(ticker_samples)

# print config
print('full_run:', full_run)
print(f'Loaded {len(all_samples)} samples from {len(csv_paths)} csvs')
print(f'Lookback Days: {lookback_days}, Lookback Weeks: {lookback_weeks}, Lookback Months: {lookback_months}')

# Prepare datasets
X_daily, X_weekly, X_monthly, y = [], [], [], []
for sample in all_samples:
    X_daily.append(sample[0])
    X_weekly.append(sample[1])
    X_monthly.append(sample[2])
    y.append(sample[3])

X_daily = np.array(X_daily)
X_weekly = np.array(X_weekly)
X_monthly = np.array(X_monthly)
y = np.array(y)

# Train-test split
(X_daily_train, X_daily_test, 
 X_weekly_train, X_weekly_test, 
 X_monthly_train, X_monthly_test, 
 y_train, y_test) = train_test_split(X_daily, X_weekly, X_monthly, y, test_size=0.2, random_state=42)

# Model architecture
daily_input = Input(shape=(lookback_days, 13))
weekly_input = Input(shape=(lookback_weeks, 13))
monthly_input = Input(shape=(lookback_months, 13))

daily_lstm = LSTM(64, return_sequences=False)(daily_input)
weekly_lstm = LSTM(64, return_sequences=False)(weekly_input)
monthly_lstm = LSTM(64, return_sequences=False)(monthly_input)

combined = Concatenate()([daily_lstm, weekly_lstm, monthly_lstm])
if full_run:
    dense = Dense(128, activation='relu')(combined)
    dense = Dense(64, activation='relu')(dense)
    dense = Dense(32, activation='sigmoid')(dense)
else:
    dense = Dense(64, activation='relu')(combined)
    dense = Dense(32, activation='sigmoid')(dense)
output = Dense(prediction_days * 4)(dense)
output = Reshape((prediction_days, 4))(output)

model = Model(inputs=[daily_input, weekly_input, monthly_input], outputs=output)
model.compile(optimizer=Adam(0.001), loss='mse')

# Train model
model.fit(
    [X_daily_train, X_weekly_train, X_monthly_train],
    y_train,
    validation_data=([X_daily_test, X_weekly_test, X_monthly_test], y_test),
    epochs=100 if full_run else 10,
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