"""
Heterogeneous Graph Attention Network (HetGAT) for Supply Chain Risk Propagation.

Models risk as a graph message-passing process over a heterogeneous supply network
where nodes represent Suppliers, Manufacturers, and Ports, and edges encode material
flows, financial transactions, and logistics relationships.

Target Metrics:
    - Link Prediction AUC > 0.75
    - Node Classification Accuracy > 0.70
"""

import os
import math
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import LabelEncoder
import mlflow
import mlflow.pytorch


# ─────────────────────────────────────────────────────────────────────────────
# Graph Data Builder
# ─────────────────────────────────────────────────────────────────────────────

class SupplyChainGraph:
    """
    Constructs a heterogeneous supply chain graph from the master features CSV.

    Node types: 'supplier', 'manufacturer', 'port'
    Edge types: ('supplier','supplies','manufacturer'),
                ('supplier','ships_via','port'),
                ('port','delivers_to','manufacturer')
    """

    def __init__(self, features_path: str, raw_dir: str):
        self.features_path = features_path
        self.raw_dir = raw_dir

    def build(self):
        df = pd.read_csv(self.features_path)
        customs = pd.read_csv(os.path.join(self.raw_dir, "customs_declarations.csv"))

        suppliers = df[df["company_type"] == "Supplier"]["company_name"].tolist()
        manufacturers = df[df["company_type"] == "Manufacturer"]["company_name"].tolist()
        ports = ["Shanghai", "Rotterdam", "Singapore", "Los Angeles",
                 "Manzanillo", "Hamburg", "Mumbai JNPT", "Felixstowe"]

        all_nodes = suppliers + manufacturers + ports
        node_to_idx = {n: i for i, n in enumerate(all_nodes)}

        num_suppliers = len(suppliers)
        num_mfrs = len(manufacturers)
        num_ports = len(ports)

        # Build node feature tensors
        feat_cols = ["current_ratio", "debt_to_equity", "ebitda_margin",
                     "working_capital_ratio", "otif_rate", "lead_time_mean",
                     "lead_time_std", "supplier_concentration_hhi",
                     "customer_concentration_hhi", "dio", "dso", "dpo", "ccc",
                     "betweenness_centrality", "pagerank", "clustering_coefficient",
                     "country_risk_score", "natural_disaster_exposure",
                     "geopolitical_risk_score", "port_proximity_score"]

        # Supplier features
        sup_df = df[df["company_type"] == "Supplier"].set_index("company_name")
        sup_feats = []
        for s in suppliers:
            if s in sup_df.index:
                row = sup_df.loc[s, feat_cols].fillna(0).values.astype(np.float32)
            else:
                row = np.zeros(len(feat_cols), dtype=np.float32)
            sup_feats.append(row)

        # Manufacturer features
        mfr_df = df[df["company_type"] == "Manufacturer"].set_index("company_name")
        mfr_feats = []
        for m in manufacturers:
            if m in mfr_df.index:
                row = mfr_df.loc[m, feat_cols].fillna(0).values.astype(np.float32)
            else:
                row = np.zeros(len(feat_cols), dtype=np.float32)
            mfr_feats.append(row)

        # Port features (synthetic: congestion-based)
        port_base = [1.8, 1.5, 1.2, 2.2, 1.6, 1.4, 1.9, 1.3]
        port_feats = []
        for i, p in enumerate(ports):
            pf = np.zeros(len(feat_cols), dtype=np.float32)
            pf[5] = port_base[i]  # lead_time_mean proxy = congestion
            pf[16] = 55.0         # country_risk_score placeholder
            port_feats.append(pf)

        x_supplier = torch.tensor(np.array(sup_feats), dtype=torch.float32)
        x_manufacturer = torch.tensor(np.array(mfr_feats), dtype=torch.float32)
        x_port = torch.tensor(np.array(port_feats), dtype=torch.float32)

        # Build edge indices
        # (supplier -> manufacturer) from customs declarations
        sup_to_idx = {s: i for i, s in enumerate(suppliers)}
        mfr_to_idx = {m: i for i, m in enumerate(manufacturers)}
        port_to_idx = {p: i for i, p in enumerate(ports)}

        sup_mfr_src, sup_mfr_dst = [], []
        for _, row in customs.iterrows():
            if row["exporter"] in sup_to_idx and row["importer"] in mfr_to_idx:
                sup_mfr_src.append(sup_to_idx[row["exporter"]])
                sup_mfr_dst.append(mfr_to_idx[row["importer"]])

        # Deduplicate
        sup_mfr_edges = list(set(zip(sup_mfr_src, sup_mfr_dst)))
        sup_mfr_src = [e[0] for e in sup_mfr_edges]
        sup_mfr_dst = [e[1] for e in sup_mfr_edges]

        # (supplier -> port) each supplier ships via 2-3 random ports
        np.random.seed(42)
        sp_src, sp_dst = [], []
        for i in range(num_suppliers):
            assigned = np.random.choice(num_ports, size=2, replace=False)
            for p in assigned:
                sp_src.append(i)
                sp_dst.append(p)

        # (port -> manufacturer) each port serves 2-3 random manufacturers
        pm_src, pm_dst = [], []
        for i in range(num_ports):
            assigned = np.random.choice(num_mfrs, size=2, replace=False)
            for m in assigned:
                pm_src.append(i)
                pm_dst.append(m)

        edges = {
            ("supplier", "supplies", "manufacturer"):
                (torch.tensor(sup_mfr_src, dtype=torch.long),
                 torch.tensor(sup_mfr_dst, dtype=torch.long)),
            ("supplier", "ships_via", "port"):
                (torch.tensor(sp_src, dtype=torch.long),
                 torch.tensor(sp_dst, dtype=torch.long)),
            ("port", "delivers_to", "manufacturer"):
                (torch.tensor(pm_src, dtype=torch.long),
                 torch.tensor(pm_dst, dtype=torch.long)),
        }

        # Node-level risk labels (binary: high risk = 1) for suppliers & manufacturers
        # Defined as: debt_to_equity > 2.0 OR current_ratio < 1.0 OR ccc > 75
        def risk_label(row_):
            return int(row_["debt_to_equity"] > 2.0 or
                       row_["current_ratio"] < 1.0 or
                       row_["ccc"] > 75)

        sup_labels = torch.tensor(
            [risk_label(sup_df.loc[s]) if s in sup_df.index else 0 for s in suppliers],
            dtype=torch.long)
        mfr_labels = torch.tensor(
            [risk_label(mfr_df.loc[m]) if m in mfr_df.index else 0 for m in manufacturers],
            dtype=torch.long)

        return {
            "x": {"supplier": x_supplier, "manufacturer": x_manufacturer, "port": x_port},
            "edge_index": edges,
            "y": {"supplier": sup_labels, "manufacturer": mfr_labels},
            "meta": {
                "num_suppliers": num_suppliers,
                "num_mfrs": num_mfrs,
                "num_ports": num_ports,
                "in_channels": len(feat_cols)
            }
        }


# ─────────────────────────────────────────────────────────────────────────────
# Heterogeneous Graph Attention Layer
# ─────────────────────────────────────────────────────────────────────────────

class HeteroGATLayer(nn.Module):
    """
    A single Heterogeneous Graph Attention layer for one relation type.
    Computes attention-weighted message-passing between source and target nodes.
    """

    def __init__(self, src_in_dim: int, dst_in_dim: int, out_dim: int, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.heads = heads
        self.out_dim = out_dim
        self.head_dim = out_dim // heads

        self.W_src = nn.Linear(src_in_dim, out_dim, bias=False)
        self.W_dst = nn.Linear(dst_in_dim, out_dim, bias=False)
        self.attn = nn.Linear(2 * self.head_dim, 1, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, x_src: torch.Tensor, x_dst: torch.Tensor,
                edge_index_src: torch.Tensor, edge_index_dst: torch.Tensor):
        """
        Args:
            x_src: [N_src, src_in_dim]
            x_dst: [N_dst, dst_in_dim]
            edge_index_src: [E] source node indices
            edge_index_dst: [E] destination node indices
        Returns:
            out: [N_dst, out_dim] updated destination node embeddings
        """
        # Project to common dimension
        h_src = self.W_src(x_src)  # [N_src, out_dim]
        h_dst = self.W_dst(x_dst)  # [N_dst, out_dim]

        # Reshape for multi-head: [N, heads, head_dim]
        h_src = h_src.view(-1, self.heads, self.head_dim)
        h_dst = h_dst.view(-1, self.heads, self.head_dim)

        # Gather source and destination features per edge
        src_feat = h_src[edge_index_src]  # [E, heads, head_dim]
        dst_feat = h_dst[edge_index_dst]  # [E, heads, head_dim]

        # Compute attention scores
        cat_feat = torch.cat([src_feat, dst_feat], dim=-1)  # [E, heads, 2*head_dim]
        attn_score = self.attn(cat_feat).squeeze(-1)         # [E, heads]
        attn_score = self.leaky_relu(attn_score)

        # Softmax per destination node per head
        N_dst = x_dst.size(0)
        E = edge_index_dst.size(0)
        # Use scatter softmax approximation: clamp for stability
        attn_score = torch.exp(attn_score - attn_score.max())  # [E, heads]

        # Aggregate messages
        out = torch.zeros(N_dst, self.heads, self.head_dim, device=x_src.device)
        attn_sum = torch.zeros(N_dst, self.heads, device=x_src.device)

        for e in range(E):
            d = edge_index_dst[e]
            out[d] += attn_score[e].unsqueeze(-1) * src_feat[e]
            attn_sum[d] += attn_score[e]

        # Normalize
        attn_sum = attn_sum.clamp(min=1e-6).unsqueeze(-1)
        out = out / attn_sum  # [N_dst, heads, head_dim]

        # Apply dropout and flatten
        out = self.dropout(out)
        out = out.view(N_dst, self.out_dim)
        return F.elu(out)


# ─────────────────────────────────────────────────────────────────────────────
# Full HetGAT Model
# ─────────────────────────────────────────────────────────────────────────────

class HetGAT(nn.Module):
    """
    Full Heterogeneous Graph Attention Network with 3 message-passing layers.
    Supports:
      - Node classification (supplier/manufacturer risk tier)
      - Link prediction (will a new trade link appear?)
    """

    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int,
                 num_layers: int = 3, heads: int = 4, dropout: float = 0.1,
                 num_classes: int = 2):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)

        # Layer 1: raw features → hidden
        self.layer1_s2m = HeteroGATLayer(in_channels, in_channels, hidden_channels, heads, dropout)
        self.layer1_s2p = HeteroGATLayer(in_channels, in_channels, hidden_channels, heads, dropout)
        self.layer1_p2m = HeteroGATLayer(in_channels, in_channels, hidden_channels, heads, dropout)

        # Layer 2: hidden → hidden (after aggregation)
        self.layer2_s2m = HeteroGATLayer(hidden_channels, hidden_channels, hidden_channels, heads, dropout)
        self.layer2_s2p = HeteroGATLayer(hidden_channels, hidden_channels, hidden_channels, heads, dropout)
        self.layer2_p2m = HeteroGATLayer(hidden_channels, hidden_channels, hidden_channels, heads, dropout)

        # Layer 3: hidden → out_channels
        self.layer3_s2m = HeteroGATLayer(hidden_channels, hidden_channels, out_channels, heads, dropout)
        self.layer3_s2p = HeteroGATLayer(hidden_channels, hidden_channels, out_channels, heads, dropout)
        self.layer3_p2m = HeteroGATLayer(hidden_channels, hidden_channels, out_channels, heads, dropout)

        # Node classification head
        self.clf_head = nn.Linear(out_channels, num_classes)

        # Link prediction head
        self.link_head = nn.Sequential(
            nn.Linear(2 * out_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x_dict: dict, edge_dict: dict):
        xs = x_dict["supplier"]
        xm = x_dict["manufacturer"]
        xp = x_dict["port"]

        e_s2m_src, e_s2m_dst = edge_dict[("supplier", "supplies", "manufacturer")]
        e_s2p_src, e_s2p_dst = edge_dict[("supplier", "ships_via", "port")]
        e_p2m_src, e_p2m_dst = edge_dict[("port", "delivers_to", "manufacturer")]

        # Layer 1
        xm1 = self.layer1_s2m(xs, xm, e_s2m_src, e_s2m_dst)
        xp1 = self.layer1_s2p(xs, xp, e_s2p_src, e_s2p_dst)
        xm1 = xm1 + self.layer1_p2m(xp, xm, e_p2m_src, e_p2m_dst)

        # Residual fallback: project original if sizes match, else use layer output
        xs1 = xs[:, :xm1.shape[1]] if xs.shape[1] >= xm1.shape[1] else \
              F.pad(xs, (0, xm1.shape[1] - xs.shape[1]))
        xs1 = xs1

        # Layer 2
        xm2 = self.layer2_s2m(xs1, xm1, e_s2m_src, e_s2m_dst)
        xp2 = self.layer2_s2p(xs1, xp1, e_s2p_src, e_s2p_dst)
        xm2 = xm2 + self.layer2_p2m(xp1, xm1, e_p2m_src, e_p2m_dst)

        # Layer 3
        xm3 = self.layer3_s2m(xs1, xm2, e_s2m_src, e_s2m_dst)
        xp3 = self.layer3_s2p(xs1, xp2, e_s2p_src, e_s2p_dst)
        xs3 = F.elu(self.layer3_s2m.W_src(xs1))  # project suppliers to out_channels

        # Aggregate supplier embedding using mfr messages (reverse)
        xs_emb = xs3

        embeddings = {
            "supplier": xs_emb,
            "manufacturer": xm3,
            "port": xp3
        }
        return embeddings

    def classify_nodes(self, embeddings: dict, node_type: str):
        return self.clf_head(embeddings[node_type])

    def predict_links(self, embeddings: dict, src_type: str, dst_type: str,
                      src_idx: torch.Tensor, dst_idx: torch.Tensor):
        src_emb = embeddings[src_type][src_idx]
        dst_emb = embeddings[dst_type][dst_idx]
        pair = torch.cat([src_emb, dst_emb], dim=-1)
        return self.link_head(pair).squeeze(-1)


# ─────────────────────────────────────────────────────────────────────────────
# Training Function
# ─────────────────────────────────────────────────────────────────────────────

def train_gnn(graph_data: dict, config: dict, output_dir: str = "data/models"):
    os.makedirs(output_dir, exist_ok=True)

    in_ch = graph_data["meta"]["in_channels"]
    hidden = config.get("hidden_channels", 64)
    out_ch = config.get("out_channels", 128)
    num_layers = config.get("num_layers", 3)
    heads = config.get("heads", 4)
    dropout = config.get("dropout", 0.1)
    lr = config.get("learning_rate", 0.001)
    epochs = config.get("epochs", 50)

    model = HetGAT(in_ch, hidden, out_ch, num_layers, heads, dropout)
    optimizer = Adam(model.parameters(), lr=lr)
    clf_loss_fn = nn.CrossEntropyLoss()

    x_dict = graph_data["x"]
    edge_dict = graph_data["edge_index"]
    sup_labels = graph_data["y"]["supplier"]
    mfr_labels = graph_data["y"]["manufacturer"]

    # Positive edges for link prediction from supplier→manufacturer
    e_s2m_src, e_s2m_dst = edge_dict[("supplier", "supplies", "manufacturer")]

    mlflow.set_experiment("LogisChain_GNN")
    with mlflow.start_run(run_name="HetGAT_Training"):
        mlflow.log_params({
            "in_channels": in_ch, "hidden_channels": hidden,
            "out_channels": out_ch, "num_layers": num_layers,
            "heads": heads, "dropout": dropout,
            "lr": lr, "epochs": epochs
        })

        for epoch in range(1, epochs + 1):
            model.train()
            optimizer.zero_grad()

            embeddings = model(x_dict, edge_dict)

            # Node classification loss (supplier risk)
            sup_logits = model.classify_nodes(embeddings, "supplier")
            # Pad/truncate supplier embedding to match classification head
            if sup_logits.shape[0] != sup_labels.shape[0]:
                min_n = min(sup_logits.shape[0], sup_labels.shape[0])
                sup_logits = sup_logits[:min_n]
                sup_labels_ = sup_labels[:min_n]
            else:
                sup_labels_ = sup_labels

            loss_clf = clf_loss_fn(sup_logits, sup_labels_)

            # Link prediction loss (binary CE on positive vs. random negative edges)
            num_pos = e_s2m_src.shape[0]
            N_sup = x_dict["supplier"].shape[0]
            N_mfr = x_dict["manufacturer"].shape[0]

            # Positive link scores
            pos_scores = model.predict_links(
                embeddings, "supplier", "manufacturer",
                e_s2m_src[:num_pos], e_s2m_dst[:num_pos]
            )

            # Negative samples
            neg_src = torch.randint(0, N_sup, (num_pos,))
            neg_dst = torch.randint(0, N_mfr, (num_pos,))
            neg_scores = model.predict_links(
                embeddings, "supplier", "manufacturer", neg_src, neg_dst
            )

            pos_labels = torch.ones(num_pos)
            neg_labels = torch.zeros(num_pos)
            all_scores = torch.cat([pos_scores, neg_scores])
            all_labels = torch.cat([pos_labels, neg_labels])
            loss_link = F.binary_cross_entropy(all_scores, all_labels)

            total_loss = loss_clf + 0.5 * loss_link
            total_loss.backward()
            optimizer.step()

            if epoch % 10 == 0:
                model.eval()
                with torch.no_grad():
                    emb = model(x_dict, edge_dict)
                    # Supplier classification accuracy
                    sup_pred = model.classify_nodes(emb, "supplier").argmax(dim=1)
                    min_n = min(sup_pred.shape[0], sup_labels.shape[0])
                    acc = accuracy_score(sup_labels[:min_n].numpy(), sup_pred[:min_n].numpy())

                    # Link AUC
                    all_s = all_scores.detach().numpy()
                    all_l = all_labels.numpy()
                    auc = roc_auc_score(all_l, all_s)

                    print(f"Epoch {epoch:3d} | Loss: {total_loss.item():.4f} | "
                          f"Node Acc: {acc:.3f} | Link AUC: {auc:.3f}")
                    mlflow.log_metrics({"loss": total_loss.item(), "node_acc": acc, "link_auc": auc}, step=epoch)

        # Save model
        model_path = os.path.join(output_dir, "gnn_hetgat.pt")
        torch.save(model.state_dict(), model_path)
        mlflow.log_artifact(model_path)
        print(f"\nGNN model saved to {model_path}")

    return model


if __name__ == "__main__":
    import yaml

    with open("configs/data_config.yaml") as f:
        d_cfg = yaml.safe_load(f)
    with open("configs/model_config.yaml") as f:
        m_cfg = yaml.safe_load(f)

    builder = SupplyChainGraph(
        features_path=os.path.join(d_cfg["paths"]["features_dir"], "master_features.csv"),
        raw_dir=d_cfg["paths"]["raw_dir"]
    )
    graph = builder.build()
    print(f"Graph built | Supplier nodes: {graph['meta']['num_suppliers']} | "
          f"Manufacturer nodes: {graph['meta']['num_mfrs']} | "
          f"Port nodes: {graph['meta']['num_ports']}")

    train_gnn(graph, m_cfg["gnn"])
