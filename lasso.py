from sklearn.linear_model import Lasso
import numpy as np

# Assuming X_train is your training data with shape (samples, time_steps, features)
X_train_reshaped = X_train.reshape(-1, X_train.shape[2])  # Flatten time dimension
y_train_reshaped = y_train.flatten()  # Flatten target if necessary

# Apply LASSO Regression
lasso = Lasso(alpha=0.01)  # Adjust alpha to control feature selection
lasso.fit(X_train_reshaped, y_train_reshaped)

# Get selected features
selected_features = np.where(lasso.coef_ != 0)[0]
print(f"Selected feature indices: {selected_features}")

# Reduce dataset dimensions based on selected features
X_train_reduced = X_train[:, :, selected_features]  # Keep only important features
X_val_reduced = X_val[:, :, selected_features]
X_test_reduced = X_test[:, :, selected_features]

feature_count_reduced = len(selected_features)  # Update feature count


daily_input = Input(shape=(lookback_days, feature_count_reduced))
weekly_input = Input(shape=(lookback_weeks, feature_count_reduced))
monthly_input = Input(shape=(lookback_months, feature_count_reduced))

rnn_units = 64 if full_run else 32

daily = LSTM(rnn_units, return_sequences=False)(daily_input)
weekly = LSTM(rnn_units, return_sequences=False)(weekly_input)
monthly = LSTM(rnn_units, return_sequences=False)(monthly_input)

layer = Concatenate()([daily, weekly, monthly])

if full_run:
    layer = Dense(rnn_units, activation='relu')(layer)
layer = Dense(32, activation='relu')(layer)
layer = Dense(prediction_days * 4)(layer)
layer = Reshape((prediction_days, 4))(layer)

model = Model(inputs=[daily_input, weekly_input, monthly_input], outputs=layer)
model.compile(optimizer=Adam(0.001), loss='mse', metrics=['accuracy'])
