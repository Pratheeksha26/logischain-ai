"""
Stacking Ensemble Meta-Learner for Trade Finance Default Prediction.

Integrates outputs from the GNN, TCN, Transformer, XGBoost, and Survival
models into a Level-1 LightGBM meta-learner using 5-fold cross-validation
stacking.

Target Metrics:
    - Gini Coefficient > 0.55 (minimum), target > 0.65
    - Expected Calibration Error (ECE) < 0.03
"""

import os
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.calibration import calibration_curve
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow


# ─────────────────────────────────────────────────────────────────────────────
# Expected Calibration Error
# ─────────────────────────────────────────────────────────────────────────────

def compute_ece(y_true, y_prob, n_bins: int = 10):
    """
    Compute Expected Calibration Error (ECE).

    ECE = sum_{b=1}^{B} (n_b / N) * |acc_b - conf_b|

    where acc_b is the accuracy and conf_b is the average predicted
    probability in bin b.
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total = len(y_true)

    for i in range(n_bins):
        mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])
        n_b = mask.sum()
        if n_b > 0:
            acc_b = y_true[mask].mean()
            conf_b = y_prob[mask].mean()
            ece += (n_b / total) * abs(acc_b - conf_b)

    return ece


def compute_gini(y_true, y_prob):
    """Gini coefficient = 2 * AUC - 1."""
    auc = roc_auc_score(y_true, y_prob)
    return 2 * auc - 1


# ─────────────────────────────────────────────────────────────────────────────
# Stacking Data Assembly
# ─────────────────────────────────────────────────────────────────────────────

def assemble_stacking_features(features_dir: str = "data/features",
                               models_dir: str = "data/models"):
    """
    Assemble Level-0 base learner predictions into a stacking feature matrix.

    In production, each base model would produce out-of-fold predictions.
    Here we simulate the stacking matrix from the master feature catalog
    to demonstrate the ensemble architecture.
    """
    df = pd.read_csv(os.path.join(features_dir, "master_features.csv"))
    np.random.seed(42)

    n = len(df)

    # Simulated Level-0 predictions from each base model
    # In practice these would be actual out-of-fold predictions
    risk_score = (
        (df["debt_to_equity"].fillna(1.0) / 3.0) +
        (1.0 / (df["current_ratio"].fillna(1.5) + 0.01)) +
        (df["ccc"].fillna(50) / 100.0) +
        (1.0 - df["otif_rate"].fillna(0.9))
    )
    risk_norm = (risk_score - risk_score.min()) / (risk_score.max() - risk_score.min() + 1e-8)

    # GNN node embedding risk scores (simulated)
    gnn_risk = np.clip(risk_norm + np.random.normal(0, 0.05, n), 0, 1)

    # TCN forecast deviation score (simulated)
    tcn_deviation = np.clip(0.3 + 0.4 * risk_norm + np.random.normal(0, 0.08, n), 0, 1)

    # Transformer delay probability (simulated)
    transformer_delay = np.clip(0.2 + 0.5 * risk_norm + np.random.normal(0, 0.06, n), 0, 1)

    # XGBoost default probability (simulated)
    xgb_default = np.clip(0.15 + 0.55 * risk_norm + np.random.normal(0, 0.04, n), 0, 1)

    # Survival hazard rate (simulated)
    surv_hazard = np.clip(0.25 + 0.45 * risk_norm + np.random.normal(0, 0.07, n), 0, 1)

    # Additional raw features to pass through
    raw_features = df[["current_ratio", "debt_to_equity", "ccc",
                       "otif_rate", "lead_time_std",
                       "supplier_concentration_hhi"]].fillna(0).values

    # Stack Level-0 predictions + select raw features
    X_stack = np.column_stack([
        gnn_risk, tcn_deviation, transformer_delay,
        xgb_default, surv_hazard, raw_features
    ])

    feature_names = [
        "gnn_risk", "tcn_deviation", "transformer_delay",
        "xgb_default", "surv_hazard",
        "current_ratio", "debt_to_equity", "ccc",
        "otif_rate", "lead_time_std", "supplier_hhi"
    ]

    # Binary default label
    y = ((df["debt_to_equity"] > 2.0) |
         (df["current_ratio"] < 1.0) |
         (df["ccc"] > 75)).astype(int).values

    return X_stack.astype(np.float32), y, feature_names


# ─────────────────────────────────────────────────────────────────────────────
# LightGBM Stacking Meta-Learner
# ─────────────────────────────────────────────────────────────────────────────

def train_ensemble(config: dict = None, features_dir: str = "data/features",
                   models_dir: str = "data/models", output_dir: str = "data/models"):
    os.makedirs(output_dir, exist_ok=True)

    if config is None:
        config = {
            "meta_learner": "LightGBM",
            "num_folds": 5,
            "learning_rate": 0.05,
            "n_estimators": 200
        }

    X, y, feature_names = assemble_stacking_features(features_dir, models_dir)
    n_folds = config.get("num_folds", 5)
    print(f"Stacking ensemble: X={X.shape}, y_pos_rate={y.mean():.3f}")

    # Cross-validated stacking predictions
    oof_preds = np.zeros(len(y))
    fold_metrics = []

    # Use StratifiedKFold since the dataset is small
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    mlflow.set_experiment("LogisChain_Ensemble")
    with mlflow.start_run(run_name="LightGBM_Stacking"):
        mlflow.log_params({
            "meta_learner": config["meta_learner"],
            "n_folds": n_folds,
            "learning_rate": config.get("learning_rate", 0.05),
            "n_estimators": config.get("n_estimators", 200)
        })

        lgb_params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "learning_rate": config.get("learning_rate", 0.05),
            "n_estimators": config.get("n_estimators", 200),
            "max_depth": 4,
            "num_leaves": 15,
            "min_child_samples": 3,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "verbose": -1
        }

        final_model = None
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            model = lgb.LGBMClassifier(**lgb_params)
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.log_evaluation(period=0)]
            )

            val_preds = model.predict_proba(X_val)[:, 1]
            oof_preds[val_idx] = val_preds

            try:
                fold_auc = roc_auc_score(y_val, val_preds)
                fold_gini = 2 * fold_auc - 1
            except ValueError:
                fold_auc = 0.5
                fold_gini = 0.0

            fold_metrics.append({"fold": fold, "auc": fold_auc, "gini": fold_gini})
            print(f"  Fold {fold} | AUC: {fold_auc:.4f} | Gini: {fold_gini:.4f}")
            final_model = model

        # Overall out-of-fold metrics
        try:
            overall_auc = roc_auc_score(y, oof_preds)
        except ValueError:
            overall_auc = 0.5
        overall_gini = compute_gini(y, oof_preds) if overall_auc > 0.5 else 0.0
        overall_ece = compute_ece(y, oof_preds)
        overall_brier = brier_score_loss(y, oof_preds)

        print(f"\n{'='*50}")
        print(f"Overall OOF Results:")
        print(f"  AUC:   {overall_auc:.4f}")
        print(f"  Gini:  {overall_gini:.4f}")
        print(f"  ECE:   {overall_ece:.4f}")
        print(f"  Brier: {overall_brier:.4f}")

        mlflow.log_metrics({
            "oof_auc": overall_auc,
            "oof_gini": overall_gini,
            "oof_ece": overall_ece,
            "oof_brier": overall_brier
        })

        # Feature importance from meta-learner
        if final_model is not None:
            importance = final_model.feature_importances_
            imp_df = pd.DataFrame({
                "feature": feature_names,
                "importance": importance
            }).sort_values("importance", ascending=False)
            print(f"\nMeta-Learner Feature Importance:")
            print(imp_df.to_string(index=False))

            # Save importance plot
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.barh(imp_df["feature"], imp_df["importance"], color="#3498db")
            ax.set_xlabel("Importance")
            ax.set_title("Stacking Meta-Learner Feature Importance")
            ax.invert_yaxis()
            plt.tight_layout()
            imp_plot_path = os.path.join(output_dir, "ensemble_importance.png")
            plt.savefig(imp_plot_path, dpi=150)
            plt.close()
            mlflow.log_artifact(imp_plot_path)

            # Save calibration plot
            fig, ax = plt.subplots(figsize=(7, 7))
            prob_true, prob_pred = calibration_curve(y, oof_preds, n_bins=8)
            ax.plot(prob_pred, prob_true, "o-", label=f"Stacking (ECE={overall_ece:.3f})")
            ax.plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")
            ax.set_xlabel("Mean predicted probability")
            ax.set_ylabel("Fraction of positives")
            ax.set_title("Reliability Diagram (Calibration Curve)")
            ax.legend()
            plt.tight_layout()
            cal_plot_path = os.path.join(output_dir, "ensemble_calibration.png")
            plt.savefig(cal_plot_path, dpi=150)
            plt.close()
            mlflow.log_artifact(cal_plot_path)

            # Save model
            model_path = os.path.join(output_dir, "ensemble_lgbm.txt")
            final_model.booster_.save_model(model_path)
            mlflow.log_artifact(model_path)
            print(f"\nEnsemble model saved to {model_path}")

    return final_model


if __name__ == "__main__":
    import yaml

    with open("configs/model_config.yaml") as f:
        m_cfg = yaml.safe_load(f)

    train_ensemble(m_cfg.get("ensemble", {}))
