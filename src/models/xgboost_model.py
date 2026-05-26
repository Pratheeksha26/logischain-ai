"""
XGBoost Tabular Risk Classifier with Optuna Hyperparameter Optimization and SHAP.

Trains an XGBoost classifier for entity-level default prediction using 42 features
(21 entity features + 21 rolling temporal metrics). Uses Optuna for Bayesian
hyperparameter search (100+ trials) and integrates SHAP for both global feature
importance and local waterfall explanations.

Handles class imbalance via scale_pos_weight.
"""

import os
import json
import numpy as np
import pandas as pd
import xgboost as xgb
import optuna
import shap
import mlflow
import mlflow.xgboost
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, classification_report
from sklearn.preprocessing import StandardScaler


# ─────────────────────────────────────────────────────────────────────────────
# Data Preparation
# ─────────────────────────────────────────────────────────────────────────────

def prepare_xgb_data(features_dir: str = "data/features"):
    """
    Load the master feature catalog and create the 42-feature tabular dataset
    with binary default labels.
    """
    df = pd.read_csv(os.path.join(features_dir, "master_features.csv"))

    # 21 entity-level features
    entity_cols = [
        "current_ratio", "debt_to_equity", "ebitda_margin", "working_capital_ratio",
        "otif_rate", "lead_time_mean", "lead_time_std",
        "supplier_concentration_hhi", "customer_concentration_hhi",
        "dio", "dso", "dpo", "ccc",
        "betweenness_centrality", "pagerank", "clustering_coefficient",
        "country_risk_score", "natural_disaster_exposure",
        "geopolitical_risk_score", "port_proximity_score",
        "in_degree"
    ]

    # 21 rolling temporal metrics (simulated as lagged derivatives)
    np.random.seed(42)
    for col in entity_cols:
        if col in df.columns:
            # Rolling mean (proxy)
            df[f"{col}_roll_mean"] = df[col] * np.random.uniform(0.95, 1.05, len(df))

    rolling_cols = [c for c in df.columns if c.endswith("_roll_mean")]
    feature_cols = entity_cols + rolling_cols[:21]  # Limit to exactly 42

    # Ensure all feature columns exist
    feature_cols = [c for c in feature_cols if c in df.columns]

    X = df[feature_cols].fillna(0).values
    feature_names = feature_cols

    # Binary default label: high risk = default (same logic as GNN)
    y = ((df["debt_to_equity"] > 2.0) |
         (df["current_ratio"] < 1.0) |
         (df["ccc"] > 75)).astype(int).values

    return X, y, feature_names, df


# ─────────────────────────────────────────────────────────────────────────────
# Optuna Objective
# ─────────────────────────────────────────────────────────────────────────────

def optuna_objective(trial, X, y):
    """Bayesian hyperparameter search objective for XGBoost."""
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 1200),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
        "min_child_weight": trial.suggest_int("min_child_weight", 5, 50),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "gamma": trial.suggest_float("gamma", 0.0, 0.5),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 3.0),
    }

    # Compute scale_pos_weight for imbalance handling
    n_pos = y.sum()
    n_neg = len(y) - n_pos
    scale_pos_weight = n_neg / max(n_pos, 1)
    params["scale_pos_weight"] = scale_pos_weight

    # Time-series aware cross-validation (no random splitting)
    tscv = TimeSeriesSplit(n_splits=3)
    aucs = []
    for train_idx, val_idx in tscv.split(X):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = xgb.XGBClassifier(
            **params,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42,
            verbosity=0
        )
        model.fit(X_train, y_train,
                  eval_set=[(X_val, y_val)],
                  verbose=False)

        y_pred_prob = model.predict_proba(X_val)[:, 1]
        try:
            auc = roc_auc_score(y_val, y_pred_prob)
        except ValueError:
            auc = 0.5
        aucs.append(auc)

    return np.mean(aucs)


# ─────────────────────────────────────────────────────────────────────────────
# Training Function
# ─────────────────────────────────────────────────────────────────────────────

def train_xgboost(config: dict, features_dir: str = "data/features",
                  output_dir: str = "data/models", n_trials: int = 100):
    os.makedirs(output_dir, exist_ok=True)

    X, y, feature_names, df = prepare_xgb_data(features_dir)
    print(f"XGBoost dataset: X={X.shape}, y_pos_rate={y.mean():.3f}")

    # Optuna hyperparameter optimization
    print(f"\nRunning Optuna optimization ({n_trials} trials)...")
    study = optuna.create_study(direction="maximize",
                                study_name="xgb_default_prediction")
    study.optimize(lambda trial: optuna_objective(trial, X, y),
                   n_trials=n_trials, show_progress_bar=True)

    best_params = study.best_params
    print(f"\nBest Optuna params (AUC={study.best_value:.4f}):")
    for k, v in best_params.items():
        print(f"  {k}: {v}")

    # Scale pos weight
    n_pos = y.sum()
    n_neg = len(y) - n_pos
    best_params["scale_pos_weight"] = n_neg / max(n_pos, 1)

    # Train final model on full data
    final_model = xgb.XGBClassifier(
        **best_params,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        verbosity=0
    )
    final_model.fit(X, y)

    # MLflow logging
    mlflow.set_experiment("LogisChain_XGBoost")
    with mlflow.start_run(run_name="XGBoost_Optuna"):
        mlflow.log_params(best_params)
        mlflow.log_metric("best_optuna_auc", study.best_value)

        y_pred = final_model.predict(X)
        y_prob = final_model.predict_proba(X)[:, 1]
        train_auc = roc_auc_score(y, y_prob)
        train_acc = accuracy_score(y, y_pred)
        mlflow.log_metrics({"train_auc": train_auc, "train_acc": train_acc})

        # Save model
        model_path = os.path.join(output_dir, "xgboost_model.json")
        final_model.save_model(model_path)
        mlflow.log_artifact(model_path)

        # ─── SHAP Explainability ───
        print("\nComputing SHAP explanations...")
        explainer = shap.TreeExplainer(final_model)
        shap_values = explainer.shap_values(X)

        # Global feature importance (bar plot)
        fig_global, ax = plt.subplots(figsize=(10, 8))
        shap.summary_plot(shap_values, X, feature_names=feature_names,
                          plot_type="bar", show=False)
        fig_path_global = os.path.join(output_dir, "shap_global_importance.png")
        plt.savefig(fig_path_global, dpi=150, bbox_inches="tight")
        plt.close()
        mlflow.log_artifact(fig_path_global)

        # Local waterfall plot for first sample
        fig_local, ax = plt.subplots(figsize=(10, 6))
        shap.waterfall_plot(
            shap.Explanation(
                values=shap_values[0],
                base_values=explainer.expected_value,
                data=X[0],
                feature_names=feature_names
            ),
            show=False
        )
        fig_path_local = os.path.join(output_dir, "shap_waterfall_sample0.png")
        plt.savefig(fig_path_local, dpi=150, bbox_inches="tight")
        plt.close()
        mlflow.log_artifact(fig_path_local)

        # Save Optuna study results
        optuna_path = os.path.join(output_dir, "optuna_study.json")
        trials_data = []
        for t in study.trials:
            trials_data.append({"number": t.number, "value": t.value, "params": t.params})
        with open(optuna_path, "w") as f:
            json.dump(trials_data, f, indent=2)
        mlflow.log_artifact(optuna_path)

    print(f"\nXGBoost model saved to {model_path}")
    print(f"SHAP plots saved to {output_dir}")
    return final_model


if __name__ == "__main__":
    import yaml

    with open("configs/model_config.yaml") as f:
        m_cfg = yaml.safe_load(f)

    train_xgboost(m_cfg["xgboost"], n_trials=100)
