"""
Temporal Convolutional Network (TCN) for Multi-Horizon Demand & Throughput Forecasting.

Implements dilated causal convolutions with dilation factors [1, 2, 4, 8, 16, 32, 64]
producing a receptive field of 128 days. Outputs distributional forecasts via quantile
regression at the 10th, 50th, and 90th percentiles.

Target Metrics:
    - MAPE < 12% on 30-day forecast horizon
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
import mlflow


# ─────────────────────────────────────────────────────────────────────────────
# Time Series Dataset
# ─────────────────────────────────────────────────────────────────────────────

class TimeSeriesDataset(Dataset):
    """
    Sliding-window dataset for port throughput and freight rate forecasting.
    Creates (input_chunk, target_chunk) pairs using a rolling window.
    """

    def __init__(self, series: np.ndarray, input_len: int = 128, output_len: int = 30):
        self.series = series.astype(np.float32)
        self.input_len = input_len
        self.output_len = output_len
        self.total_len = input_len + output_len

    def __len__(self):
        return max(0, len(self.series) - self.total_len + 1)

    def __getitem__(self, idx):
        x = self.series[idx: idx + self.input_len]
        y = self.series[idx + self.input_len: idx + self.total_len]
        return torch.tensor(x).unsqueeze(-1), torch.tensor(y)


# ─────────────────────────────────────────────────────────────────────────────
# Causal Convolution Block
# ─────────────────────────────────────────────────────────────────────────────

class CausalConv1dBlock(nn.Module):
    """
    A single dilated causal convolution block with:
      - Causal padding (left-only)
      - Weight normalization
      - ReLU activation
      - Dropout
      - Residual connection with 1x1 projection if channels change
    """

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int = 3, dilation: int = 1, dropout: float = 0.2):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation

        self.conv1 = nn.utils.weight_norm(
            nn.Conv1d(in_channels, out_channels, kernel_size,
                      dilation=dilation, padding=0)
        )
        self.conv2 = nn.utils.weight_norm(
            nn.Conv1d(out_channels, out_channels, kernel_size,
                      dilation=dilation, padding=0)
        )
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # Residual projection
        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        """x: [B, C, T]"""
        res = self.residual(x)

        # First causal conv
        out = F.pad(x, (self.padding, 0))
        out = self.conv1(out)
        out = self.relu(out)
        out = self.dropout(out)

        # Second causal conv
        out = F.pad(out, (self.padding, 0))
        out = self.conv2(out)
        out = self.relu(out)
        out = self.dropout(out)

        # Residual add
        return self.relu(out + res)


# ─────────────────────────────────────────────────────────────────────────────
# TCN Model
# ─────────────────────────────────────────────────────────────────────────────

class TCNModel(nn.Module):
    """
    Temporal Convolutional Network with 7 layers of dilated causal convolutions
    (dilations: 1, 2, 4, 8, 16, 32, 64) and multi-quantile prediction heads.

    Architecture:
        Input [B, T, 1] → Transpose → 7x CausalConv1dBlock → GlobalPool → QuantileHeads
    """

    def __init__(self, input_channels: int = 1, num_filters: int = 64,
                 kernel_size: int = 3, dropout: float = 0.2,
                 output_len: int = 30, quantiles: list = None):
        super().__init__()
        self.output_len = output_len
        self.quantiles = quantiles or [0.1, 0.5, 0.9]

        # Dilated causal convolution stack: dilations [1, 2, 4, 8, 16, 32, 64]
        dilations = [1, 2, 4, 8, 16, 32, 64]
        layers = []
        in_ch = input_channels
        for d in dilations:
            layers.append(CausalConv1dBlock(in_ch, num_filters, kernel_size, d, dropout))
            in_ch = num_filters
        self.tcn_stack = nn.Sequential(*layers)

        # Prediction heads: one linear layer per quantile
        self.quantile_heads = nn.ModuleList([
            nn.Linear(num_filters, output_len) for _ in self.quantiles
        ])

    def forward(self, x):
        """
        Args:
            x: [B, T, 1] input time series
        Returns:
            dict mapping quantile → [B, output_len] predictions
        """
        # Transpose to [B, C, T] for Conv1d
        out = x.permute(0, 2, 1)
        out = self.tcn_stack(out)

        # Global average pooling over time dimension
        out = out.mean(dim=-1)  # [B, num_filters]

        # Generate quantile forecasts
        predictions = {}
        for i, q in enumerate(self.quantiles):
            predictions[q] = self.quantile_heads[i](out)

        return predictions


# ─────────────────────────────────────────────────────────────────────────────
# Quantile Loss
# ─────────────────────────────────────────────────────────────────────────────

def quantile_loss(predictions: dict, targets: torch.Tensor) -> torch.Tensor:
    """
    Pinball (quantile) loss function.

    For quantile q:
        L_q(y, ŷ) = q * max(y - ŷ, 0) + (1 - q) * max(ŷ - y, 0)
    """
    total_loss = 0.0
    for q, pred in predictions.items():
        errors = targets - pred
        loss = torch.max(q * errors, (q - 1) * errors)
        total_loss += loss.mean()
    return total_loss / len(predictions)


# ─────────────────────────────────────────────────────────────────────────────
# Data Preparation
# ─────────────────────────────────────────────────────────────────────────────

def prepare_tcn_data(raw_dir: str = "data/raw", input_len: int = 128,
                     output_len: int = 30, port_code: str = "CNSHA"):
    """Load port throughput time series and create train/val splits."""
    df = pd.read_csv(os.path.join(raw_dir, "port_throughput.csv"))
    df["date"] = pd.to_datetime(df["date"])

    # Filter to a single port and resample daily
    port_df = df[df["port_code"] == port_code].sort_values("date")
    series = port_df["throughput_teu"].values.astype(np.float32)

    # Normalize
    mean_ = series.mean()
    std_ = series.std() + 1e-8
    series_norm = (series - mean_) / std_

    # Temporal split: 80% train, 20% val (no random shuffle to prevent leakage)
    split_idx = int(len(series_norm) * 0.8)
    train_series = series_norm[:split_idx]
    val_series = series_norm[split_idx - input_len:]  # overlap for context

    train_ds = TimeSeriesDataset(train_series, input_len, output_len)
    val_ds = TimeSeriesDataset(val_series, input_len, output_len)

    return train_ds, val_ds, mean_, std_


# ─────────────────────────────────────────────────────────────────────────────
# Training Function
# ─────────────────────────────────────────────────────────────────────────────

def train_tcn(config: dict, raw_dir: str = "data/raw",
              output_dir: str = "data/models"):
    os.makedirs(output_dir, exist_ok=True)

    input_len = config.get("input_chunk_length", 128)
    output_len = config.get("output_chunk_length", 30)
    num_filters = config.get("num_filters", 64)
    kernel_size = config.get("kernel_size", 3)
    dropout = config.get("dropout", 0.2)
    lr = config.get("learning_rate", 0.001)
    epochs = config.get("epochs", 100)
    quantiles = config.get("quantiles", [0.1, 0.5, 0.9])

    train_ds, val_ds, mean_, std_ = prepare_tcn_data(raw_dir, input_len, output_len)
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)

    model = TCNModel(
        input_channels=1, num_filters=num_filters,
        kernel_size=kernel_size, dropout=dropout,
        output_len=output_len, quantiles=quantiles
    )
    optimizer = Adam(model.parameters(), lr=lr)

    mlflow.set_experiment("LogisChain_TCN")
    with mlflow.start_run(run_name="TCN_Training"):
        mlflow.log_params({
            "input_len": input_len, "output_len": output_len,
            "num_filters": num_filters, "kernel_size": kernel_size,
            "dropout": dropout, "lr": lr, "epochs": epochs
        })

        for epoch in range(1, epochs + 1):
            model.train()
            epoch_loss = 0.0
            n_batches = 0

            for x_batch, y_batch in train_loader:
                optimizer.zero_grad()
                preds = model(x_batch)
                loss = quantile_loss(preds, y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)

            # Validation
            if epoch % 10 == 0:
                model.eval()
                val_losses = []
                mape_vals = []
                with torch.no_grad():
                    for x_val, y_val in val_loader:
                        preds = model(x_val)
                        v_loss = quantile_loss(preds, y_val)
                        val_losses.append(v_loss.item())

                        # Compute MAPE on median (50th percentile) forecast
                        median_pred = preds[0.5]
                        # De-normalize
                        y_actual = y_val * std_ + mean_
                        y_pred = median_pred * std_ + mean_
                        mape = torch.mean(
                            torch.abs((y_actual - y_pred) / (y_actual.abs() + 1e-8))
                        ).item() * 100
                        mape_vals.append(mape)

                avg_val = np.mean(val_losses)
                avg_mape = np.mean(mape_vals)
                print(f"Epoch {epoch:3d} | Train Loss: {avg_loss:.4f} | "
                      f"Val Loss: {avg_val:.4f} | MAPE: {avg_mape:.1f}%")
                mlflow.log_metrics({
                    "train_loss": avg_loss,
                    "val_loss": avg_val,
                    "mape": avg_mape
                }, step=epoch)

        # Save model
        model_path = os.path.join(output_dir, "tcn_model.pt")
        torch.save(model.state_dict(), model_path)
        mlflow.log_artifact(model_path)
        print(f"\nTCN model saved to {model_path}")

    return model


if __name__ == "__main__":
    import yaml

    with open("configs/model_config.yaml") as f:
        m_cfg = yaml.safe_load(f)

    train_tcn(m_cfg["tcn"])
