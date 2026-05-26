"""
Transformer Shipment Risk Encoder for Anomaly Prediction.

Models each shipment as a sequence of 8 logistics event tokens and applies
multi-head self-attention (4+ heads) to combine spatial (lat/lon), temporal
(timestamps), and contextual (weather, port congestion) embeddings for
predicting delay probability, damage risk, and customs hold likelihood.

Target Metrics:
    - AUC > 0.80 for delay prediction
    - Brier Score < 0.18 for calibration
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, brier_score_loss
import mlflow


# ─────────────────────────────────────────────────────────────────────────────
# Shipment Event Dataset
# ─────────────────────────────────────────────────────────────────────────────

# Event vocabulary: 8 canonical logistics milestones
EVENT_VOCAB = [
    "BOOKING_CONFIRMED",
    "CONTAINER_LOADED",
    "VESSEL_DEPARTED",
    "TRANSHIPMENT_ARRIVAL",
    "TRANSHIPMENT_DEPARTURE",
    "DESTINATION_PORT_ARRIVAL",
    "CUSTOMS_CLEARED",
    "FINAL_DELIVERY"
]
EVENT_TO_IDX = {e: i for i, e in enumerate(EVENT_VOCAB)}
NUM_EVENTS = len(EVENT_VOCAB)


class ShipmentEventDataset(Dataset):
    """
    Builds event token sequences from the raw shipments data.
    Each shipment produces a sequence of 8 events with temporal, spatial,
    and contextual features attached to each event step.
    """

    def __init__(self, shipments_path: str, ports_path: str):
        df_ship = pd.read_csv(shipments_path)
        df_ports = pd.read_csv(ports_path)

        # Build port congestion lookup (average congestion per port)
        port_cong = df_ports.groupby("port_code")["congestion_index"].mean().to_dict()

        self.samples = []
        self.labels = []

        for _, row in df_ship.iterrows():
            booking_date = pd.to_datetime(row["booking_date"])
            transit = row["actual_transit_days"]
            delay = row["delay_days"]
            value = row["value_usd"]
            weight = row["weight_tons"]

            orig_code = row["origin_port_code"]
            dest_code = row["destination_port_code"]
            orig_cong = port_cong.get(orig_code, 1.5)
            dest_cong = port_cong.get(dest_code, 1.5)

            # Simulate event-level features for each of 8 milestones
            # Features per event: [event_idx, day_offset, lat, lon, congestion,
            #                      value_norm, weight_norm, speed_proxy]
            features = []
            for i, event in enumerate(EVENT_VOCAB):
                day_offset = (transit / NUM_EVENTS) * i
                # Interpolated position between origin/dest
                frac = i / (NUM_EVENTS - 1)

                # Lat/lon approximation (mock interpolation)
                lat = 31.0 * (1 - frac) + 51.0 * frac + np.random.normal(0, 2)
                lon = 121.0 * (1 - frac) + 4.0 * frac + np.random.normal(0, 5)

                # Congestion at this stage
                cong = orig_cong * (1 - frac) + dest_cong * frac

                feat = [
                    float(i),                           # Event index
                    day_offset / 60.0,                  # Normalized day offset
                    lat / 90.0,                         # Normalized latitude
                    lon / 180.0,                        # Normalized longitude
                    cong / 3.0,                         # Normalized congestion
                    min(value / 5e6, 1.0),              # Normalized value
                    min(weight / 30.0, 1.0),            # Normalized weight
                    np.random.uniform(0.5, 1.0)         # Speed proxy
                ]
                features.append(feat)

            self.samples.append(np.array(features, dtype=np.float32))

            # Multi-label: [is_delayed, is_damaged, has_document_discrepancy]
            self.labels.append([
                int(row["is_delayed"]),
                int(row["is_damaged"]),
                int(row["has_document_discrepancy"])
            ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return (torch.tensor(self.samples[idx]),
                torch.tensor(self.labels[idx], dtype=torch.float32))


# ─────────────────────────────────────────────────────────────────────────────
# Positional Encoding
# ─────────────────────────────────────────────────────────────────────────────

class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding for sequence positions."""

    def __init__(self, d_model: int, max_len: int = 16):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model > 1:
            pe[:, 1::2] = torch.cos(position * div_term[:d_model // 2])
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        """x: [B, seq_len, d_model]"""
        return x + self.pe[:, :x.size(1), :]


# ─────────────────────────────────────────────────────────────────────────────
# Shipment Risk Transformer
# ─────────────────────────────────────────────────────────────────────────────

class ShipmentRiskTransformer(nn.Module):
    """
    Multi-head self-attention Transformer Encoder that processes shipment event
    sequences and outputs anomaly risk probabilities.

    Architecture:
        Input Projection → Positional Encoding → N Transformer Encoder Layers
        → [CLS] pooling → Multi-task Classification Heads
    """

    def __init__(self, input_dim: int = 8, d_model: int = 64,
                 num_heads: int = 4, num_layers: int = 3,
                 dim_feedforward: int = 128, dropout: float = 0.1,
                 num_tasks: int = 3):
        super().__init__()

        # Input projection
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=NUM_EVENTS)

        # Transformer encoder stack
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Multi-task classification heads
        self.task_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, 1),
                nn.Sigmoid()
            ) for _ in range(num_tasks)
        ])

    def forward(self, x):
        """
        Args:
            x: [B, seq_len, input_dim] event feature sequences
        Returns:
            preds: [B, num_tasks] probability predictions
        """
        # Project input to d_model
        h = self.input_proj(x)      # [B, seq_len, d_model]
        h = self.pos_enc(h)

        # Transformer encoding
        h = self.encoder(h)         # [B, seq_len, d_model]

        # Pool: use mean of all event token embeddings
        h_pooled = h.mean(dim=1)    # [B, d_model]

        # Multi-task predictions
        preds = torch.cat([head(h_pooled) for head in self.task_heads], dim=-1)
        return preds  # [B, num_tasks]


# ─────────────────────────────────────────────────────────────────────────────
# Training Function
# ─────────────────────────────────────────────────────────────────────────────

def train_transformer(config: dict, raw_dir: str = "data/raw",
                      output_dir: str = "data/models"):
    os.makedirs(output_dir, exist_ok=True)

    d_model = config.get("d_model", 64)
    num_heads = config.get("num_heads", 4)
    num_layers = config.get("num_layers", 3)
    dim_ff = config.get("dim_feedforward", 128)
    dropout = config.get("dropout", 0.1)
    lr = config.get("learning_rate", 0.0005)
    epochs = config.get("epochs", 40)

    # Build dataset
    dataset = ShipmentEventDataset(
        os.path.join(raw_dir, "shipments.csv"),
        os.path.join(raw_dir, "port_throughput.csv")
    )

    # Temporal split: 80/20
    n = len(dataset)
    split = int(n * 0.8)
    train_ds = torch.utils.data.Subset(dataset, range(split))
    val_ds = torch.utils.data.Subset(dataset, range(split, n))

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)

    model = ShipmentRiskTransformer(
        input_dim=8, d_model=d_model, num_heads=num_heads,
        num_layers=num_layers, dim_feedforward=dim_ff,
        dropout=dropout, num_tasks=3
    )
    optimizer = Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCELoss()

    mlflow.set_experiment("LogisChain_Transformer")
    with mlflow.start_run(run_name="ShipmentRisk_Training"):
        mlflow.log_params({
            "d_model": d_model, "num_heads": num_heads,
            "num_layers": num_layers, "dim_feedforward": dim_ff,
            "dropout": dropout, "lr": lr, "epochs": epochs
        })

        for epoch in range(1, epochs + 1):
            model.train()
            epoch_loss = 0.0
            n_batches = 0

            for x_batch, y_batch in train_loader:
                optimizer.zero_grad()
                preds = model(x_batch)
                loss = loss_fn(preds, y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)

            # Validation every 5 epochs
            if epoch % 5 == 0:
                model.eval()
                all_preds = []
                all_labels = []
                with torch.no_grad():
                    for x_val, y_val in val_loader:
                        p = model(x_val)
                        all_preds.append(p.numpy())
                        all_labels.append(y_val.numpy())

                all_preds = np.vstack(all_preds)
                all_labels = np.vstack(all_labels)

                # Delay prediction metrics (task 0)
                try:
                    delay_auc = roc_auc_score(all_labels[:, 0], all_preds[:, 0])
                except ValueError:
                    delay_auc = 0.5
                delay_brier = brier_score_loss(all_labels[:, 0], all_preds[:, 0])

                print(f"Epoch {epoch:3d} | Loss: {avg_loss:.4f} | "
                      f"Delay AUC: {delay_auc:.3f} | Brier: {delay_brier:.4f}")
                mlflow.log_metrics({
                    "train_loss": avg_loss,
                    "delay_auc": delay_auc,
                    "delay_brier": delay_brier
                }, step=epoch)

        # Save model
        model_path = os.path.join(output_dir, "transformer_risk.pt")
        torch.save(model.state_dict(), model_path)
        mlflow.log_artifact(model_path)
        print(f"\nTransformer model saved to {model_path}")

    return model


if __name__ == "__main__":
    import yaml

    with open("configs/model_config.yaml") as f:
        m_cfg = yaml.safe_load(f)

    train_transformer(m_cfg["transformer"])
