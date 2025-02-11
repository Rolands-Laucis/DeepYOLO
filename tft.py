import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import glob
from pytorch_forecasting.models import TemporalFusionTransformer
from pytorch_forecasting.data.timeseries import TimeSeriesDataSet
from pytorch_forecasting.metrics import QuantileLoss

# Configuration
LOOKBACK_DAYS = 21
LOOKBACK_WEEKS = 6
FEATURE_COUNT = 13
PREDICTION_DAYS = 5
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001
MAX_ENCODER_LENGTH = LOOKBACK_DAYS + LOOKBACK_WEEKS
MAX_PREDICTION_LENGTH = PREDICTION_DAYS

# Dummy Data (Replace with actual dataset loading)
num_samples = 1000
time_idx = np.arange(num_samples)
X_daily = torch.randn(num_samples, LOOKBACK_DAYS, FEATURE_COUNT, dtype=torch.float32)
X_weekly = torch.randn(num_samples, LOOKBACK_WEEKS, FEATURE_COUNT, dtype=torch.float32)
y = torch.randn(num_samples, PREDICTION_DAYS, 4, dtype=torch.float32)  # Predicting Open, Close, High, Low

# Combine daily and weekly data
X_combined = torch.cat((X_daily, X_weekly), dim=1).numpy().reshape(num_samples, -1)
y_combined = y.numpy().reshape(num_samples, -1)  # Combine all targets into a single column

data_dict = {"time_idx": time_idx, "group_id": np.zeros(num_samples), "target": y_combined[:, 0]}  # Use the first target as an example for now
for i in range(X_combined.shape[1]):
    data_dict[f"X_{i}"] = X_combined[:, i]

data_df = pd.DataFrame(data_dict)

dataset = TimeSeriesDataSet(
    data=data_df,
    time_idx="time_idx",
    target="target",
    group_ids=["group_id"],
    time_varying_known_reals=[f"X_{i}" for i in range(X_combined.shape[1])],
    max_encoder_length=MAX_ENCODER_LENGTH,
    max_prediction_length=MAX_PREDICTION_LENGTH,
)

# Prepare DataLoader
train_dataloader = dataset.to_dataloader(train=True, batch_size=BATCH_SIZE, shuffle=True)

# Define TFT Model
class StockTemporalFusionTransformer(nn.Module):
    def __init__(self, dataset):
        super().__init__()
        self.tft = TemporalFusionTransformer.from_dataset(
            dataset,
            hidden_size=32,
            lstm_layers=1,
            attention_head_size=4,
            dropout=0.1,
            loss=QuantileLoss(),
        )

    def forward(self, x):
        return self.tft(x)

# Initialize Model
model = StockTemporalFusionTransformer(dataset)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Custom Loss Function to Handle Multi-Target
def custom_loss(y_pred, y_true):
    y_pred = y_pred["prediction"]  # Extract prediction tensor
    y_true = y_true[0]  # Extract target tensor from the tuple
    y_true = y_true.view_as(y_pred)  # Reshape y_true to match y_pred
    return nn.MSELoss()(y_pred, y_true)

# Training Loop
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for batch in train_dataloader:
        X_batch, y_batch = batch
        optimizer.zero_grad()
        y_pred = model(X_batch)
        loss = custom_loss(y_pred, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss/len(train_dataloader):.4f}")

print("Training complete!")