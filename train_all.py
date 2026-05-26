import os
import sys
import yaml
import warnings
warnings.filterwarnings('ignore')

# Ensure current directory is in system path
sys.path.append(os.path.abspath("."))

def main():
    import sys
    sys.stdout.reconfigure(encoding='utf-8')
    print("="*60)
    print("🚀 Starting Unified Model Training Pipeline for LogisChain AI")
    print("="*60)
    
    with open("configs/data_config.yaml") as f:
        data_cfg = yaml.safe_load(f)
    with open("configs/model_config.yaml") as f:
        model_cfg = yaml.safe_load(f)
        
    output_dir = "data/models"
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Train GNN (HetGAT)
    print("\n--- 1. Training HetGAT GNN ---")
    from src.models.gnn import SupplyChainGraph, train_gnn
    builder = SupplyChainGraph(
        features_path=os.path.join(data_cfg['paths']['features_dir'], 'master_features.csv'),
        raw_dir=data_cfg['paths']['raw_dir']
    )
    graph_data = builder.build()
    print(f"Graph: {graph_data['meta']['num_suppliers']} suppliers, "
          f"{graph_data['meta']['num_mfrs']} manufacturers, "
          f"{graph_data['meta']['num_ports']} ports")
    train_gnn(graph_data, model_cfg['gnn'], output_dir=output_dir)
    
    # 2. Train TCN
    print("\n--- 2. Training TCN Forecaster ---")
    from src.models.tcn import train_tcn
    train_tcn(
        model_cfg['tcn'],
        raw_dir=data_cfg['paths']['raw_dir'],
        output_dir=output_dir
    )
    
    # 3. Train Transformer
    print("\n--- 3. Training Transformer Shipment Risk Encoder ---")
    from src.models.transformer import train_transformer
    train_transformer(
        model_cfg['transformer'],
        raw_dir=data_cfg['paths']['raw_dir'],
        output_dir=output_dir
    )
    
    # 4. Train XGBoost
    print("\n--- 4. Training XGBoost Baseline + Optuna + SHAP ---")
    from src.models.xgboost_model import train_xgboost
    train_xgboost(
        model_cfg['xgboost'],
        features_dir=data_cfg['paths']['features_dir'],
        output_dir=output_dir,
        n_trials=30  # Lowering to 30 trials to balance training time with optimal hyperparameter search
    )
    
    # 5. Train Survival Models (Cox PH & DeepSurv)
    print("\n--- 5. Training Survival Models ---")
    from src.models.survival import train_coxph, train_deepsurv
    cph_model, c1 = train_coxph(
        features_dir=data_cfg['paths']['features_dir'],
        output_dir=output_dir
    )
    print(f'Cox PH C-index: {c1:.4f}')
    
    ds_model, c2 = train_deepsurv(
        features_dir=data_cfg['paths']['features_dir'],
        output_dir=output_dir,
        epochs=100
    )
    print(f'DeepSurv C-index: {c2:.4f}')
    
    # 6. Train Stacking Ensemble
    print("\n--- 6. Training Stacking Ensemble ---")
    from src.models.ensemble import train_ensemble
    train_ensemble(
        config=model_cfg.get('ensemble', {}),
        features_dir=data_cfg['paths']['features_dir'],
        models_dir=output_dir,
        output_dir=output_dir
    )
    
    print("\n" + "="*60)
    print("🎉 All Models Trained and Saved Successfully!")
    print("="*60)

if __name__ == "__main__":
    main()
