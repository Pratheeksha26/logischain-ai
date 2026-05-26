"""
Survival Analysis Models for Supply Chain Default Time-to-Event Prediction.

Implements two approaches:
  1. Cox Proportional Hazards (CoxPH) using lifelines
  2. DeepSurv - a deep neural network extension of CoxPH using PyTorch

Models the continuous time until a supplier/manufacturer defaults or a critical
supply chain link fails, based on financial and operational covariates.

Target Metrics:
    - Concordance Index (C-index) > 0.80
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
import mlflow


# ─────────────────────────────────────────────────────────────────────────────
# Survival Data Preparation
# ─────────────────────────────────────────────────────────────────────────────

def prepare_survival_data(features_dir: str = "data/features"):
    """
    Construct a survival dataset from the master feature catalog.

    Duration: simulated as days until a financial distress event (high CCC,
              low current ratio, or high debt-to-equity breach).
    Event:    1 if the entity experienced distress within the observation
              window, 0 if right-censored.
    """
    df = pd.read_csv(os.path.join(features_dir, "master_features.csv"))

    np.random.seed(42)

    # Simulate duration (time-to-event in days)
    # Higher risk entities default sooner
    risk_score = (
        (df["debt_to_equity"].fillna(1.0) / 3.0) +
        (1.0 / (df["current_ratio"].fillna(1.5) + 0.01)) +
        (df["ccc"].fillna(50) / 100.0) +
        (1.0 - df["otif_rate"].fillna(0.9))
    )
    # Normalize to [0, 1]
    risk_norm = (risk_score - risk_score.min()) / (risk_score.max() - risk_score.min() + 1e-8)

    # Duration inversely proportional to risk: high risk → short duration
    base_duration = 365.0  # 1-year observation window
    duration = base_duration * (1 - 0.7 * risk_norm) + np.random.normal(0, 30, len(df))
    duration = np.clip(duration, 30, 365)

    # Event indicator: entities with duration < 300 days are "observed defaults"
    event = (duration < 300).astype(int)

    # Feature columns for survival modeling
    covariate_cols = [
        "current_ratio", "debt_to_equity", "ebitda_margin", "working_capital_ratio",
        "otif_rate", "lead_time_mean", "lead_time_std",
        "supplier_concentration_hhi", "customer_concentration_hhi",
        "dio", "dso", "dpo", "ccc",
        "betweenness_centrality", "pagerank", "clustering_coefficient",
        "country_risk_score", "natural_disaster_exposure",
        "geopolitical_risk_score", "port_proximity_score"
    ]
    covariate_cols = [c for c in covariate_cols if c in df.columns]

    X = df[covariate_cols].fillna(0).values.astype(np.float32)

    # Build a lifelines-compatible DataFrame
    surv_df = df[covariate_cols].fillna(0).copy()
    surv_df["duration"] = duration
    surv_df["event"] = event

    return X, duration.astype(np.float32), event.astype(np.float32), covariate_cols, surv_df


# ─────────────────────────────────────────────────────────────────────────────
# Cox Proportional Hazards (lifelines)
# ─────────────────────────────────────────────────────────────────────────────

def train_coxph(features_dir: str = "data/features", output_dir: str = "data/models"):
    """Train a classical Cox PH model using lifelines."""
    os.makedirs(output_dir, exist_ok=True)

    _, _, _, covariate_cols, surv_df = prepare_survival_data(features_dir)

    # Filter out constant columns from surv_df to prevent lifelines convergence errors
    # (since the dataset is small and synthetic, some features have 0 variance)
    non_constant_cols = []
    for col in covariate_cols:
        if surv_df[col].std() > 1e-4:
            non_constant_cols.append(col)
        else:
            print(f"Skipping constant feature in CoxPH: {col}")

    fit_cols = non_constant_cols + ["duration", "event"]
    fit_df = surv_df[fit_cols].copy()

    # Increase penalizer to stabilize convergence on highly correlated/small sample data
    cph = CoxPHFitter(penalizer=0.5)
    
    try:
        cph.fit(fit_df, duration_col="duration", event_col="event")
        c_index = cph.concordance_index_
        print(f"\nCox PH Model Results:")
        print(f"  Concordance Index: {c_index:.4f}")
        cph.print_summary()
    except Exception as e:
        print(f"Warning: CoxPH fitting failed with error: {e}. Using a simplified model.")
        # Fallback to a very simple model with robust features
        fallback_cols = ["current_ratio", "debt_to_equity", "ccc", "duration", "event"]
        fallback_cols = [c for c in fallback_cols if c in fit_df.columns]
        cph = CoxPHFitter(penalizer=1.0)
        cph.fit(fit_df[fallback_cols], duration_col="duration", event_col="event")
        c_index = cph.concordance_index_
        print(f"\nFallback Cox PH Model Results:")
        print(f"  Concordance Index: {c_index:.4f}")

    # Save
    model_path = os.path.join(output_dir, "coxph_model.pkl")
    import pickle
    with open(model_path, "wb") as f:
        pickle.dump(cph, f)

    mlflow.set_experiment("LogisChain_Survival")
    with mlflow.start_run(run_name="CoxPH"):
        mlflow.log_metric("c_index", c_index)
        mlflow.log_artifact(model_path)

    print(f"Cox PH model saved to {model_path}")
    return cph, c_index


# ─────────────────────────────────────────────────────────────────────────────
# DeepSurv Neural Network
# ─────────────────────────────────────────────────────────────────────────────

class SurvivalDataset(Dataset):
    """PyTorch dataset for survival data."""

    def __init__(self, X, duration, event):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.duration = torch.tensor(duration, dtype=torch.float32)
        self.event = torch.tensor(event, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.duration[idx], self.event[idx]


class DeepSurv(nn.Module):
    """
    Deep Neural Network extension of Cox Proportional Hazards.

    Architecture:
        Input → FC(hidden) → BN → SELU → Dropout → FC(hidden) → BN → SELU
        → Dropout → FC(1) → Linear risk score

    The output is a log-risk score (log hazard ratio) that is used
    in the partial likelihood loss for training.
    """

    def __init__(self, in_features: int, hidden_dim: int = 64,
                 num_layers: int = 3, dropout: float = 0.2):
        super().__init__()

        layers = []
        current_dim = in_features
        for _ in range(num_layers):
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.SELU(),
                nn.AlphaDropout(dropout)
            ])
            current_dim = hidden_dim

        layers.append(nn.Linear(hidden_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """Returns log-risk score [B, 1]."""
        return self.network(x)


def negative_log_partial_likelihood(risk_scores, durations, events):
    """
    Cox partial likelihood loss for DeepSurv.

    For each observed event, the loss penalizes the model if the risk score
    of the event subject is not higher than those still at risk.
    """
    # Sort by duration (descending)
    sorted_idx = torch.argsort(durations, descending=True)
    risk_sorted = risk_scores[sorted_idx].squeeze()
    events_sorted = events[sorted_idx]

    # Log-sum-exp trick for numerical stability
    log_cumsum = torch.logcumsumexp(risk_sorted, dim=0)

    # Partial likelihood: sum over events
    event_mask = events_sorted.bool()
    if event_mask.sum() == 0:
        return torch.tensor(0.0, requires_grad=True)

    loss = -torch.mean(risk_sorted[event_mask] - log_cumsum[event_mask])
    return loss


def compute_c_index(risk_scores, durations, events):
    """Compute concordance index (C-statistic)."""
    risk_np = -risk_scores.detach().numpy().flatten()  # negative because higher risk = shorter duration
    dur_np = durations.numpy()
    ev_np = events.numpy()
    try:
        return concordance_index(dur_np, risk_np, ev_np)
    except Exception:
        return 0.5


# ─────────────────────────────────────────────────────────────────────────────
# DeepSurv Training
# ─────────────────────────────────────────────────────────────────────────────

def train_deepsurv(features_dir: str = "data/features",
                   output_dir: str = "data/models",
                   hidden_dim: int = 64, num_layers: int = 3,
                   dropout: float = 0.2, lr: float = 0.001,
                   epochs: int = 100):
    os.makedirs(output_dir, exist_ok=True)

    X, duration, event, cov_cols, _ = prepare_survival_data(features_dir)

    # Normalize features
    mean = X.mean(axis=0)
    std = X.std(axis=0) + 1e-8
    X_norm = (X - mean) / std

    dataset = SurvivalDataset(X_norm, duration, event)
    loader = DataLoader(dataset, batch_size=min(32, len(dataset)), shuffle=True)

    model = DeepSurv(
        in_features=X.shape[1],
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout
    )
    optimizer = Adam(model.parameters(), lr=lr)

    mlflow.set_experiment("LogisChain_Survival")
    with mlflow.start_run(run_name="DeepSurv"):
        mlflow.log_params({
            "hidden_dim": hidden_dim, "num_layers": num_layers,
            "dropout": dropout, "lr": lr, "epochs": epochs
        })

        for epoch in range(1, epochs + 1):
            model.train()
            epoch_loss = 0.0
            n_batches = 0

            for x_batch, dur_batch, ev_batch in loader:
                optimizer.zero_grad()
                risk = model(x_batch)
                loss = negative_log_partial_likelihood(risk, dur_batch, ev_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1

            if epoch % 20 == 0:
                model.eval()
                with torch.no_grad():
                    all_x = torch.tensor(X_norm, dtype=torch.float32)
                    all_risk = model(all_x)
                    c_idx = compute_c_index(
                        all_risk,
                        torch.tensor(duration),
                        torch.tensor(event)
                    )

                avg_loss = epoch_loss / max(n_batches, 1)
                print(f"Epoch {epoch:3d} | Loss: {avg_loss:.4f} | C-index: {c_idx:.4f}")
                mlflow.log_metrics({"loss": avg_loss, "c_index": c_idx}, step=epoch)

        # Final C-index
        model.eval()
        with torch.no_grad():
            all_x = torch.tensor(X_norm, dtype=torch.float32)
            all_risk = model(all_x)
            final_c = compute_c_index(all_risk, torch.tensor(duration), torch.tensor(event))

        # Save model
        model_path = os.path.join(output_dir, "deepsurv_model.pt")
        torch.save({
            "state_dict": model.state_dict(),
            "mean": mean,
            "std": std,
            "covariates": cov_cols
        }, model_path)
        mlflow.log_metric("final_c_index", final_c)
        mlflow.log_artifact(model_path)

    print(f"\nDeepSurv model saved to {model_path}")
    print(f"Final Concordance Index: {final_c:.4f}")
    return model, final_c


if __name__ == "__main__":
    print("=" * 60)
    print("Training Cox Proportional Hazards (lifelines)")
    print("=" * 60)
    cph, c1 = train_coxph()

    print("\n" + "=" * 60)
    print("Training DeepSurv Neural Network")
    print("=" * 60)
    dsurv, c2 = train_deepsurv()

    print(f"\n{'Model':<20} {'C-index':>10}")
    print("-" * 32)
    print(f"{'Cox PH':<20} {c1:>10.4f}")
    print(f"{'DeepSurv':<20} {c2:>10.4f}")
