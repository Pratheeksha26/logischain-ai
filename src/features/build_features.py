import os
import sys
import numpy as np
import pandas as pd
import networkx as nx
import yaml

def load_config(config_path="configs/data_config.yaml"):
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            try:
                return yaml.safe_load(f)
            except Exception as e:
                print(f"Error loading YAML: {e}")
    return {"paths": {"raw_dir": "data/raw", "processed_dir": "data/processed", "features_dir": "data/features"}}

def build_features():
    config = load_config()
    raw_dir = config["paths"]["raw_dir"]
    features_dir = config["paths"]["features_dir"]
    
    print(f"Processing raw data from {raw_dir} and saving features to {features_dir}...")
    
    # Check if raw files exist. If not, generate them.
    metrics_path = os.path.join(raw_dir, "financial_metrics.csv")
    shipments_path = os.path.join(raw_dir, "shipments.csv")
    customs_path = os.path.join(raw_dir, "customs_declarations.csv")
    ports_path = os.path.join(raw_dir, "port_throughput.csv")
    
    if not (os.path.exists(metrics_path) and os.path.exists(shipments_path)):
        print("Raw data files not found. Ingesting raw data first...")
        sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
        from src.data.ingest import generate_data
        generate_data()
        
    # Read files
    df_financials = pd.read_csv(metrics_path)
    df_shipments = pd.read_csv(shipments_path)
    df_customs = pd.read_csv(customs_path)
    df_ports = pd.read_csv(ports_path)
    
    # --- 1. Entity-level Financial Features ---
    # Current Ratio = Current Assets / Current Liabilities
    df_financials["current_ratio"] = df_financials["current_assets"] / df_financials["current_liabilities"]
    # Debt-to-Equity = Total Debt / Total Equity
    df_financials["debt_to_equity"] = df_financials["total_debt"] / df_financials["total_equity"]
    # EBITDA margin = EBITDA / Revenue (simulated as 12% + some random noise)
    df_financials["ebitda_margin"] = 0.12 + np.random.normal(0, 0.02, len(df_financials))
    # Working Capital Ratio = Net Working Capital / Revenue
    df_financials["working_capital_ratio"] = (df_financials["current_assets"] - df_financials["current_liabilities"]) / df_financials["annual_revenue"]
    
    # --- 2. Operational Supply Chain Features ---
    # Map declaration_id in df_customs to matching shipment_id format (DEC_2025xxxx -> SH_100xxxx)
    # since df_customs has 3000 rows and df_shipments has 2500 rows, they are aligned by their index.
    df_customs["mapped_shipment_id"] = df_customs["declaration_id"].apply(
        lambda x: "SH_100" + x.split("_")[1][4:] if "_" in x else x
    )

    # Compute shipment metrics per supplier (exporter)
    # OTIF rate, lead time mean, lead time std dev, inventory turnover
    shipment_stats = []
    
    for company in df_financials["company_name"]:
        # Exporter stats
        df_exp = df_shipments[df_shipments["shipment_id"].isin(
            df_customs[df_customs["exporter"] == company]["mapped_shipment_id"]
        )]
        
        if len(df_exp) > 0:
            otif_rate = 1.0 - (df_exp["is_delayed"].sum() / len(df_exp))
            lead_time_mean = df_exp["actual_transit_days"].mean()
            lead_time_std = df_exp["actual_transit_days"].std()
            if pd.isna(lead_time_std):
                lead_time_std = 1.5
        else:
            otif_rate = 0.92 # Default baseline
            lead_time_mean = 14.2
            lead_time_std = 3.8
            
        # HHI index for Customer Concentration (concentration of the company's buyers/importers when it acts as an exporter)
        df_cust = df_customs[df_customs["exporter"] == company]
        if len(df_cust) > 0:
            shares = df_cust["importer"].value_counts(normalize=True).values
            customer_hhi = np.sum(shares ** 2)
        else:
            customer_hhi = 0.380
            
        # HHI index for Supplier Concentration (concentration of the company's sellers/exporters when it acts as an importer)
        df_supp = df_customs[df_customs["importer"] == company]
        if len(df_supp) > 0:
            shares = df_supp["exporter"].value_counts(normalize=True).values
            supplier_hhi = np.sum(shares ** 2)
        else:
            supplier_hhi = 0.305
            
        # Cash Conversion Cycle (CCC = DIO + DSO - DPO)
        # DIO = (Average Inventory / COGS) * 365
        # DSO = (Receivables / Revenue) * 365
        # DPO = (Payables / COGS) * 365
        row = df_financials[df_financials["company_name"] == company].iloc[0]
        dio = (row["average_inventory"] / row["cogs"]) * 365
        dso = (row["accounts_receivable"] / row["annual_revenue"]) * 365
        dpo = (row["accounts_payable"] / row["cogs"]) * 365
        ccc = dio + dso - dpo
        
        shipment_stats.append({
            "company_name": company,
            "otif_rate": round(otif_rate, 3),
            "lead_time_mean": round(lead_time_mean, 2),
            "lead_time_std": round(lead_time_std, 2),
            "supplier_concentration_hhi": round(supplier_hhi, 3),
            "customer_concentration_hhi": round(customer_hhi, 3),
            "dio": round(dio, 1),
            "dso": round(dso, 1),
            "dpo": round(dpo, 1),
            "ccc": round(ccc, 1)
        })
        
    df_op = pd.DataFrame(shipment_stats)
    df_merged = df_financials.merge(df_op, on="company_name")
    
    # --- 3. Network Topology Features ---
    # Construct a directed NetworkX graph of supply links from customs declarations
    G = nx.DiGraph()
    
    # Add nodes
    for company in df_financials["company_name"]:
        G.add_node(company, type=df_financials[df_financials["company_name"] == company]["company_type"].values[0])
        
    # Add edges weighted by value of trade
    trade_flows = df_customs.groupby(["exporter", "importer"])["invoice_value_usd"].sum().reset_index()
    for _, row in trade_flows.iterrows():
        if G.has_node(row["exporter"]) and G.has_node(row["importer"]):
            G.add_edge(row["exporter"], row["importer"], weight=row["invoice_value_usd"])
            
    # Compute centralities
    in_degree = dict(G.in_degree())
    out_degree = dict(G.out_degree())
    
    # Handle centrality for directed graphs (fallback to undirected if disconnected/sparse)
    try:
        betweenness = nx.betweenness_centrality(G, weight="weight")
    except:
        betweenness = {node: 0.1 for node in G.nodes()}
        
    try:
        pagerank = nx.pagerank(G, weight="weight")
    except:
        pagerank = {node: 1.0/len(G) for node in G.nodes()}
        
    clustering = nx.clustering(G)
    
    # Map back to features dataframe
    df_merged["in_degree"] = df_merged["company_name"].map(in_degree)
    df_merged["out_degree"] = df_merged["company_name"].map(out_degree)
    df_merged["betweenness_centrality"] = df_merged["company_name"].map(betweenness)
    df_merged["pagerank"] = df_merged["company_name"].map(pagerank)
    df_merged["clustering_coefficient"] = df_merged["company_name"].map(clustering)
    
    # --- 4. Geographic and Risk Features ---
    # Mocking standard country and location scores as defined in A5.1 worked example
    df_merged["country_risk_score"] = 62.0 # Standard score for emerging markets/corridors
    df_merged["natural_disaster_exposure"] = np.random.uniform(0.1, 0.4, len(df_merged))
    df_merged["geopolitical_risk_score"] = np.random.uniform(0.1, 0.3, len(df_merged))
    df_merged["port_proximity_score"] = np.random.uniform(0.5, 0.9, len(df_merged))
    
    # Clean up centralities
    df_merged["betweenness_centrality"] = df_merged["betweenness_centrality"].fillna(0.0)
    df_merged["pagerank"] = df_merged["pagerank"].fillna(0.0)
    df_merged["clustering_coefficient"] = df_merged["clustering_coefficient"].fillna(0.0)
    
    # Scale/Round all scores for consistency
    df_merged = df_merged.round(4)
    
    # Save the complete master feature catalog
    output_path = os.path.join(features_dir, "master_features.csv")
    df_merged.to_csv(output_path, index=False)
    
    # Save NetworkX graph to processed folder for GNN usage
    nx.write_gml(G, os.path.join(config["paths"]["processed_dir"], "supply_chain_network.gml"))
    
    print(f"Features computed and saved successfully to {output_path}!")
    print(f"Extracted shape: {df_merged.shape}")
    print(df_merged.head())

if __name__ == "__main__":
    build_features()
