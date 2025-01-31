import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiScaleTemporalFusionTransformer(nn.Module):
    def __init__(self, input_features, num_time_steps, output_steps, num_heads, ff_dim, num_layers, dropout=0.1):
        super(MultiScaleTemporalFusionTransformer, self).__init__()
        
        # Input parameters
        self.input_features = input_features  # Number of features per time step (OHLC + technical indicators)
        self.num_time_steps = num_time_steps  # Number of historical time steps (e.g., 20 candles)
        self.output_steps = output_steps      # Number of future candles to predict (e.g., 5)
        self.num_heads = num_heads            # Number of attention heads
        self.ff_dim = ff_dim                  # Feedforward dimension
        self.num_layers = num_layers          # Number of transformer layers
        self.dropout = dropout                # Dropout rate

        # Feature extraction
        self.tcn = nn.Sequential(
            nn.Conv1d(input_features, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=4, dilation=4),
            nn.ReLU(),
        )

        # Time embedding for the current day of the year
        self.time_embedding = nn.Linear(1, 64)  # Embedding for the current day of the year (scalar)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=64, nhead=num_heads, dim_feedforward=ff_dim, dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output head for OHLC predictions
        self.output_head = nn.Linear(64, output_steps * 4)  # Predicts OHLC for `output_steps` candles

    def forward(self, x, current_day):
        # x: (batch_size, num_time_steps, input_features) - OHLC + technical indicators
        # current_day: (batch_size, 1) - Scalar representing the current day of the year (normalized to [1, 2])

        # Feature extraction
        x = x.permute(0, 2, 1)  # (batch_size, input_features, num_time_steps)
        x = self.tcn(x)         # (batch_size, 64, num_time_steps)
        x = x.permute(0, 2, 1)  # (batch_size, num_time_steps, 64)

        # Time embedding
        current_day = current_day.unsqueeze(1)  # (batch_size, 1, 1)
        time_embed = self.time_embedding(current_day)  # (batch_size, 1, 64)
        time_embed = time_embed.expand(-1, self.num_time_steps, -1)  # (batch_size, num_time_steps, 64)

        # Add time embedding to features
        x = x + time_embed

        # Transformer encoder
        x = x.permute(1, 0, 2)  # (num_time_steps, batch_size, 64) for Transformer
        x = self.transformer_encoder(x)
        x = x.permute(1, 0, 2)  # (batch_size, num_time_steps, 64)

        # Output head
        x = self.output_head(x[:, -1, :])  # Use the last time step's output
        x = x.view(-1, self.output_steps, 4)  # Reshape to (batch_size, output_steps, 4) for OHLC

        return x

# Example usage
if __name__ == "__main__":
    # Hyperparameters
    batch_size = 32
    num_time_steps = 20  # 20 candles
    input_features = 10  # OHLC + 6 technical indicators
    output_steps = 5     # Predict next 5 1-day candles
    num_heads = 4
    ff_dim = 128
    num_layers = 2
    dropout = 0.1

    # Initialize model
    model = MultiScaleTemporalFusionTransformer(
        input_features=input_features,
        num_time_steps=num_time_steps,
        output_steps=output_steps,
        num_heads=num_heads,
        ff_dim=ff_dim,
        num_layers=num_layers,
        dropout=dropout,
    )

    # Example input
    x = torch.randn(batch_size, num_time_steps, input_features)  # OHLC + technical indicators
    current_day = torch.rand(batch_size, 1) * 1.0 + 1.0  # Random scalar in [1, 2] for current day of the year

    # Forward pass
    predictions = model(x, current_day)
    print(predictions.shape)  # Should be (batch_size, output_steps, 4)