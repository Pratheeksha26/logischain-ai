import os
import sys
import random
import numpy as np
import pandas as pd
import yaml
from datetime import datetime, timedelta

def load_config(config_path="configs/data_config.yaml"):
    """Loads configuration from YAML or returns defaults."""
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            try:
                return yaml.safe_load(f)
            except Exception as e:
                print(f"Error loading YAML config: {e}. Using defaults.")
    
    # Defaults in case config file is not found
    return {
        "paths": {
            "raw_dir": "data/raw",
            "processed_dir": "data/processed",
            "features_dir": "data/features"
        },
        "simulation": {
            "random_seed": 42,
            "num_suppliers": 15,
            "num_manufacturers": 5,
            "num_logistics_providers": 3,
            "num_ports": 8,
            "num_customers": 10,
            "num_vessels": 50,
            "num_days": 365
        },
        "ports": [
            {"name": "Shanghai", "code": "CNSHA", "coordinates": [31.2304, 121.4737], "base_congestion": 1.8},
            {"name": "Rotterdam", "code": "NLRTM", "coordinates": [51.9244, 4.4777], "base_congestion": 1.5},
            {"name": "Singapore", "code": "SGSIN", "coordinates": [1.3521, 103.8198], "base_congestion": 1.2},
            {"name": "Los Angeles", "code": "USLAX", "coordinates": [34.0522, -118.2437], "base_congestion": 2.2},
            {"name": "Manzanillo", "code": "MXZLO", "coordinates": [19.0522, -104.3158], "base_congestion": 1.6}
        ],
        "modalities": [
            {"name": "Ocean", "base_cost_per_km": 0.05, "reliability": 0.88},
            {"name": "Air", "base_cost_per_km": 0.80, "reliability": 0.98},
            {"name": "Rail", "base_cost_per_km": 0.15, "reliability": 0.92},
            {"name": "Road", "base_cost_per_km": 0.25, "reliability": 0.90}
        ],
        "commodities": [
            {"code": "8528.72", "name": "Consumer Electronics", "risk_category": "Medium", "base_value": 2400000},
            {"code": "8708.29", "name": "Automotive Components", "risk_category": "High", "base_value": 1800000},
            {"code": "3004.90", "name": "Pharmaceuticals", "risk_category": "High", "base_value": 3500000},
            {"code": "7210.49", "name": "Steel Sheets", "risk_category": "Low", "base_value": 800000}
        ]
    }

def generate_data():
    config = load_config()
    sim_params = config["simulation"]
    raw_dir = config["paths"]["raw_dir"]
    
    # Ensure directories exist
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(config["paths"]["processed_dir"], exist_ok=True)
    os.makedirs(config["paths"]["features_dir"], exist_ok=True)
    
    # Set seed
    random.seed(sim_params["random_seed"])
    np.random.seed(sim_params["random_seed"])
    
    print(f"Generating synthetic supply chain and trade finance data into: {raw_dir}")
    
    # Base date
    start_date = datetime(2025, 1, 1)
    num_days = sim_params["num_days"]
    
    # 1. Company Financial Metrics (Suppliers, Manufacturers, Customers)
    companies = []
    # Generate Suppliers
    for i in range(sim_params["num_suppliers"]):
        rev = np.random.uniform(50, 200) * 1e6 # 50M to 200M
        cogs = rev * np.random.uniform(0.70, 0.80)
        inventory = cogs * np.random.uniform(0.12, 0.18) # ~45 to 65 days DIO
        receivables = rev * np.random.uniform(0.10, 0.15) # ~36 to 55 days DSO
        payables = cogs * np.random.uniform(0.11, 0.16) # ~40 to 60 days DPO
        current_assets = inventory + receivables + np.random.uniform(5, 20) * 1e6
        current_liabs = payables + np.random.uniform(5, 15) * 1e6
        equity = current_assets * np.random.uniform(0.4, 0.6)
        debt = (current_assets + np.random.uniform(20, 50) * 1e6) - equity - current_liabs
        
        companies.append({
            "company_name": f"Supplier_{i+1}",
            "company_type": "Supplier",
            "annual_revenue": rev,
            "cogs": cogs,
            "current_assets": current_assets,
            "current_liabilities": current_liabs,
            "total_debt": max(debt, 0),
            "total_equity": equity,
            "average_inventory": inventory,
            "accounts_receivable": receivables,
            "accounts_payable": payables
        })
        
    # Generate Manufacturers
    for i in range(sim_params["num_manufacturers"]):
        rev = np.random.uniform(200, 600) * 1e6 # 200M to 600M
        cogs = rev * np.random.uniform(0.65, 0.75)
        inventory = cogs * np.random.uniform(0.15, 0.22)
        receivables = rev * np.random.uniform(0.12, 0.16)
        payables = cogs * np.random.uniform(0.12, 0.18)
        current_assets = inventory + receivables + np.random.uniform(20, 50) * 1e6
        current_liabs = payables + np.random.uniform(15, 40) * 1e6
        equity = current_assets * np.random.uniform(0.4, 0.6)
        debt = (current_assets + np.random.uniform(50, 150) * 1e6) - equity - current_liabs
        
        companies.append({
            "company_name": f"Manufacturer_{i+1}",
            "company_type": "Manufacturer",
            "annual_revenue": rev,
            "cogs": cogs,
            "current_assets": current_assets,
            "current_liabilities": current_liabs,
            "total_debt": max(debt, 0),
            "total_equity": equity,
            "average_inventory": inventory,
            "accounts_receivable": receivables,
            "accounts_payable": payables
        })
        
    df_financials = pd.DataFrame(companies)
    df_financials.to_csv(os.path.join(raw_dir, "financial_metrics.csv"), index=False)
    
    # 2. Port Throughput and Congestion
    port_records = []
    ports = config["ports"]
    for day in range(num_days):
        current_date = start_date + timedelta(days=day)
        # Seasonal factor (higher congestion in Q4 shipping season)
        month = current_date.month
        season_mult = 1.25 if month in [10, 11, 12] else (0.9 if month in [5, 6, 7] else 1.0)
        
        for port in ports:
            # Random walks overlaying base congestion + seasonality
            rand_factor = np.random.normal(0, 0.15)
            congestion = max(port["base_congestion"] * season_mult + rand_factor, 0.1)
            teus = int(np.random.normal(120000, 15000) * (congestion / port["base_congestion"]))
            berth_occ = min(0.4 + 0.1 * congestion + np.random.uniform(-0.05, 0.05), 0.99)
            dwell = max(1.5 + 2.0 * congestion + np.random.normal(0, 0.5), 0.5)
            queue = int(max(congestion * 4 + np.random.normal(0, 3), 0))
            
            port_records.append({
                "date": current_date.strftime("%Y-%m-%d"),
                "port_code": port["code"],
                "port_name": port["name"],
                "throughput_teu": teus,
                "berth_occupancy": berth_occ,
                "dwell_time_days": dwell,
                "queue_size": queue,
                "congestion_index": congestion
            })
            
    df_ports = pd.DataFrame(port_records)
    df_ports.to_csv(os.path.join(raw_dir, "port_throughput.csv"), index=False)
    
    # 3. Freight Rates (spot pricing index)
    freight_records = []
    lanes = [
        "China->Europe", "China->US West Coast", "Europe->US East Coast", "China->Mexico"
    ]
    for day in range(0, num_days, 7): # Weekly rates
        current_date = start_date + timedelta(days=day)
        month = current_date.month
        season_mult = 1.30 if month in [10, 11, 12] else (0.85 if month in [4, 5, 6] else 1.0)
        
        for lane in lanes:
            base_rate = 3500 if "US" in lane else (4000 if "Europe" in lane else 2200)
            # Add random shock factor
            shock = 1.0
            if day > 140 and day < 165 and lane == "China->Europe": # Suez Blockage mock shock
                shock = 2.2 # 120% increase!
            
            rate = base_rate * season_mult * shock * np.random.uniform(0.9, 1.1)
            
            freight_records.append({
                "date": current_date.strftime("%Y-%m-%d"),
                "lane": lane,
                "container_type": "40ft FEU",
                "spot_rate": int(rate)
            })
            
    df_freight = pd.DataFrame(freight_records)
    df_freight.to_csv(os.path.join(raw_dir, "freight_rates.csv"), index=False)
    
    # 4. AIS Vessel Telemetry
    ais_records = []
    vessel_names = [f"MV_Pacific_{name}" for name in ["Star", "Wave", "Wind", "Breeze", "Storm", "Titan", "Runner", "Courier"]]
    vessels = []
    for i in range(sim_params["num_vessels"]):
        vessels.append({
            "mmsi": 200000000 + i,
            "vessel_name": vessel_names[i % len(vessel_names)] + f"_{i}",
            "carrier": random.choice(["Maersk", "MSC", "CMA CGM", "COSCO", "Hapag-Lloyd"]),
            "capacity_teu": random.choice([8000, 12000, 15000, 20000])
        })
        
    for day in range(0, num_days, 2): # Telemetry records every 2 days
        current_date = start_date + timedelta(days=day)
        
        for v in vessels:
            speed = np.random.uniform(12.0, 22.0)
            heading = np.random.uniform(0, 360)
            # Generate mock coordinates moving generally along shipping routes
            lat = np.random.uniform(-10.0, 45.0)
            lon = np.random.uniform(-120.0, 120.0)
            dest = random.choice(ports)
            cargo_weight = v["capacity_teu"] * np.random.uniform(0.6, 0.95) * 14 # 14 tons per TEU avg
            draft = np.random.uniform(10.5, 16.2)
            
            ais_records.append({
                "date": current_date.strftime("%Y-%m-%d"),
                "mmsi": v["mmsi"],
                "vessel_name": v["vessel_name"],
                "carrier": v["carrier"],
                "speed_knots": round(speed, 2),
                "heading_degrees": int(heading),
                "latitude": round(lat, 5),
                "longitude": round(lon, 5),
                "destination_port": dest["name"],
                "destination_port_code": dest["code"],
                "draft_meters": round(draft, 2),
                "cargo_weight_ktons": round(cargo_weight / 1000, 2)
            })
            
    df_ais = pd.DataFrame(ais_records)
    df_ais.to_csv(os.path.join(raw_dir, "ais_telemetry.csv"), index=False)
    
    # 5. Customs & Trade Declarations
    customs_records = []
    exporter_companies = [c["company_name"] for c in companies if c["company_type"] == "Supplier"]
    importer_companies = [c["company_name"] for c in companies if c["company_type"] == "Manufacturer"]
    
    for i in range(3000): # 3000 historical shipments/declarations
        decl_id = f"DEC_{20250000 + i}"
        commodity = random.choice(config["commodities"])
        exporter = random.choice(exporter_companies)
        importer = random.choice(importer_companies)
        val = commodity["base_value"] * np.random.uniform(0.5, 1.5)
        
        # Decide if discrepancies exist in customs docs (typical cause of trade delays)
        has_disc = 1 if np.random.rand() < 0.18 else 0 # 18% discrepancy rate
        
        customs_records.append({
            "declaration_id": decl_id,
            "declaration_date": (start_date + timedelta(days=np.random.randint(0, num_days))).strftime("%Y-%m-%d"),
            "hs_code": commodity["code"],
            "commodity_name": commodity["name"],
            "invoice_value_usd": round(val, 2),
            "exporter": exporter,
            "importer": importer,
            "origin_country": "China" if "CNSHA" in [p["code"] for p in ports] else "Other",
            "destination_country": "Mexico" if i%3==0 else ("USA" if i%3==1 else "Netherlands"),
            "has_document_discrepancy": has_disc,
            "customs_status": "Flagged" if has_disc else "Released"
        })
        
    df_customs = pd.DataFrame(customs_records)
    df_customs.to_csv(os.path.join(raw_dir, "customs_declarations.csv"), index=False)
    
    # 6. Shipments Database (combining links and events)
    shipment_records = []
    carriers = ["Maersk", "MSC", "CMA CGM", "COSCO", "Hapag-Lloyd"]
    
    for i in range(2500):
        ship_id = f"SH_{1000000 + i}"
        booking = start_date + timedelta(days=np.random.randint(0, num_days - 60))
        origin = random.choice(ports)
        dest = random.choice([p for p in ports if p["code"] != origin["code"]])
        commodity = random.choice(config["commodities"])
        carrier = random.choice(carriers)
        value = commodity["base_value"] * np.random.uniform(0.4, 1.4)
        weight_tons = np.random.uniform(5, 25)
        
        # Calculate lead times
        base_transit = 10 if dest["code"] == "SGSIN" else (22 if "MX" in dest["code"] else 28)
        # Port delays
        origin_congestion = df_ports[(df_ports["port_code"] == origin["code"]) & 
                                     (df_ports["date"] == booking.strftime("%Y-%m-%d"))]["congestion_index"].values
        orig_cong = origin_congestion[0] if len(origin_congestion) > 0 else origin["base_congestion"]
        
        dest_congestion = df_ports[(df_ports["port_code"] == dest["code"]) & 
                                   (df_ports["date"] == (booking + timedelta(days=base_transit)).strftime("%Y-%m-%d"))]["congestion_index"].values
        dest_cong = dest_congestion[0] if len(dest_congestion) > 0 else dest["base_congestion"]
        
        delay_days = (orig_cong * 1.5) + (dest_cong * 2.0) + np.random.normal(1.2, 0.8)
        # Add documentary hold delay
        has_disc = customs_records[i % len(customs_records)]["has_document_discrepancy"]
        if has_disc:
            delay_days += np.random.uniform(4.0, 10.0)
            
        actual_transit = int(base_transit + delay_days)
        actual_delivery = booking + timedelta(days=actual_transit)
        
        # Determine status
        status = "Delivered" if actual_delivery < (start_date + timedelta(days=num_days)) else "In Transit"
        is_delayed = 1 if delay_days > 5.0 else 0
        is_damaged = 1 if np.random.rand() < 0.02 else 0 # 2% damage rate
        
        shipment_records.append({
            "shipment_id": ship_id,
            "booking_date": booking.strftime("%Y-%m-%d"),
            "origin_port": origin["name"],
            "origin_port_code": origin["code"],
            "destination_port": dest["name"],
            "destination_port_code": dest["code"],
            "carrier": carrier,
            "commodity_code": commodity["code"],
            "value_usd": round(value, 2),
            "weight_tons": round(weight_tons, 2),
            "base_transit_days": base_transit,
            "delay_days": round(delay_days, 1),
            "actual_transit_days": actual_transit,
            "actual_delivery_date": actual_delivery.strftime("%Y-%m-%d"),
            "status": status,
            "is_delayed": is_delayed,
            "is_damaged": is_damaged,
            "has_document_discrepancy": has_disc
        })
        
    df_shipments = pd.DataFrame(shipment_records)
    df_shipments.to_csv(os.path.join(raw_dir, "shipments.csv"), index=False)
    
    print("Synthetic data generation finished successfully!")

if __name__ == "__main__":
    generate_data()
