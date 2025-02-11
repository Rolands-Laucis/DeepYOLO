import numpy as np
import pandas as pd
from pathlib import Path
import torch
import pytorch_lightning as pl
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data import GroupNormalizer, MultiNormalizer
from pytorch_forecasting.metrics import MAE, MultiLoss
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load dataset
path = Path('data/indicators/days/')
df = pd.read_csv(path / '0_A.csv')

# Data preprocessing
df = df.drop(columns=['Date'])
df['time_idx'] = np.arange(len(df))
df = df.ffill().bfill()

# Configure categorical group_id
df['group_id'] = 'single_series'
df['group_id'] = df['group_id'].astype('category')

# Dataset parameters
max_prediction_length = 5
max_encoder_length = 21
training_cutoff = df["time_idx"].max() - max_prediction_length

# Create TimeSeriesDataSet with proper normalization
training = TimeSeriesDataSet(
    df[lambda x: x.time_idx <= training_cutoff],
    time_idx="time_idx",
    target=["Open", "High", "Low", "Close"],
    group_ids=["group_id"],
    min_encoder_length=max_encoder_length // 2,
    max_encoder_length=max_encoder_length,
    min_prediction_length=1,
    max_prediction_length=max_prediction_length,
    static_categoricals=["group_id"],
    time_varying_known_reals=[
        "time_idx", "Date_sin",
        "28_EMA", "Stoch_RSI", "Stoch_RSI_D",
        "MACD", "MACD_Signal", "MACD_Hist",
        "SAR", "SuperTrend"
    ],
    time_varying_unknown_reals=["Open", "High", "Low", "Close"],
    target_normalizer=MultiNormalizer(
        [GroupNormalizer(groups=["group_id"], transformation=None) for _ in range(4)]
    ),
    add_relative_time_idx=True,
    add_target_scales=True,
    add_encoder_length=True,
    allow_missing_timesteps=True
)

# Create validation dataset
validation = TimeSeriesDataSet.from_dataset(training, df, predict=True, stop_randomization=True)

# Create dataloaders with num_workers=0 for stability
batch_size = 64
train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size*10, num_workers=0)

# Configure TFT model with version-compatible settings
tft = TemporalFusionTransformer.from_dataset(
    training,
    learning_rate=0.03,
    hidden_size=32,
    attention_head_size=2,
    dropout=0.1,
    hidden_continuous_size=16,
    loss=MultiLoss(metrics=[MAE() for _ in range(4)]),
    output_size=[1, 1, 1, 1],
    reduce_on_plateau_patience=3,
    optimizer="adam"
)

# Configure trainer with compatibility settings
trainer = pl.Trainer(
    max_epochs=50,
    accelerator="cpu",
    enable_model_summary=True,
    gradient_clip_val=0.1,
    callbacks=[
        EarlyStopping(monitor="val_loss", patience=5, mode="min"),
        LearningRateMonitor()
    ],
    enable_checkpointing=False,
    logger=False
)

# Train model
trainer.fit(
    tft,
    train_dataloaders=train_dataloader,
    val_dataloaders=val_dataloader
)

# Generate predictions
raw_predictions = tft.predict(val_dataloader, mode="raw")
predictions = torch.cat([x[0] for x in raw_predictions[0]]).numpy()
actuals = torch.cat([y[0] for x, (y, _) in iter(val_dataloader)]).numpy()

# Reshape data
predictions = predictions.reshape(-1, 4)
actuals = actuals.reshape(-1, 4)

# Calculate metrics
print("\nEvaluation Metrics:")
print(f"MAE: {mean_absolute_error(actuals, predictions):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(actuals, predictions)):.4f}")

# Per-feature metrics
features = ["Open", "High", "Low", "Close"]
for i, feature in enumerate(features):
    print(f"\n{feature}:")
    print(f"MAE: {mean_absolute_error(actuals[:, i], predictions[:, i]):.4f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(actuals[:, i], predictions[:, i])):.4f}")