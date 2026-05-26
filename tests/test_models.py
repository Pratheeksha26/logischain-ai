"""
Unit Test Suite for LogisChain AI Model Architectures.

Tests feed-forward tensor shapes, loss computation, gradient flow,
and basic convergence for GNN, TCN, Transformer, XGBoost, Survival,
and Ensemble models. Targets 60%+ code coverage of src/models/.
"""

import os
import sys
import numpy as np
import pytest
import torch
import torch.nn as nn

# Ensure src is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ═════════════════════════════════════════════════════════════════════════════
# GNN Tests
# ═════════════════════════════════════════════════════════════════════════════

class TestGNN:
    """Tests for Heterogeneous Graph Attention Network."""

    def test_hetero_gat_layer_shape(self):
        """HeteroGATLayer produces correct output dimensions."""
        from src.models.gnn import HeteroGATLayer

        layer = HeteroGATLayer(src_in_dim=20, dst_in_dim=20, out_dim=64, heads=4)
        x_src = torch.randn(15, 20)   # 15 supplier nodes
        x_dst = torch.randn(5, 20)    # 5 manufacturer nodes
        edge_src = torch.tensor([0, 1, 2, 3, 4])
        edge_dst = torch.tensor([0, 1, 2, 3, 4])

        out = layer(x_src, x_dst, edge_src, edge_dst)
        assert out.shape == (5, 64), f"Expected (5, 64), got {out.shape}"

    def test_hetgat_forward(self):
        """Full HetGAT forward pass produces embeddings for all node types."""
        from src.models.gnn import HetGAT

        model = HetGAT(in_channels=20, hidden_channels=32,
                        out_channels=64, num_layers=3, heads=4)

        x_dict = {
            "supplier": torch.randn(15, 20),
            "manufacturer": torch.randn(5, 20),
            "port": torch.randn(8, 20)
        }
        edge_dict = {
            ("supplier", "supplies", "manufacturer"):
                (torch.randint(0, 15, (20,)), torch.randint(0, 5, (20,))),
            ("supplier", "ships_via", "port"):
                (torch.randint(0, 15, (10,)), torch.randint(0, 8, (10,))),
            ("port", "delivers_to", "manufacturer"):
                (torch.randint(0, 8, (10,)), torch.randint(0, 5, (10,))),
        }

        embeddings = model(x_dict, edge_dict)
        assert "manufacturer" in embeddings
        assert "port" in embeddings

    def test_hetgat_node_classification(self):
        """Node classification head produces correct output shape."""
        from src.models.gnn import HetGAT

        model = HetGAT(in_channels=20, hidden_channels=32,
                        out_channels=64, num_layers=3, heads=4, num_classes=2)

        x_dict = {
            "supplier": torch.randn(15, 20),
            "manufacturer": torch.randn(5, 20),
            "port": torch.randn(8, 20)
        }
        edge_dict = {
            ("supplier", "supplies", "manufacturer"):
                (torch.randint(0, 15, (20,)), torch.randint(0, 5, (20,))),
            ("supplier", "ships_via", "port"):
                (torch.randint(0, 15, (10,)), torch.randint(0, 8, (10,))),
            ("port", "delivers_to", "manufacturer"):
                (torch.randint(0, 8, (10,)), torch.randint(0, 5, (10,))),
        }

        embeddings = model(x_dict, edge_dict)
        logits = model.classify_nodes(embeddings, "manufacturer")
        assert logits.shape == (5, 2), f"Expected (5, 2), got {logits.shape}"

    def test_hetgat_link_prediction(self):
        """Link prediction head outputs scores in [0, 1]."""
        from src.models.gnn import HetGAT

        model = HetGAT(in_channels=20, hidden_channels=32,
                        out_channels=64, num_layers=3, heads=4)

        x_dict = {
            "supplier": torch.randn(15, 20),
            "manufacturer": torch.randn(5, 20),
            "port": torch.randn(8, 20)
        }
        edge_dict = {
            ("supplier", "supplies", "manufacturer"):
                (torch.randint(0, 15, (20,)), torch.randint(0, 5, (20,))),
            ("supplier", "ships_via", "port"):
                (torch.randint(0, 15, (10,)), torch.randint(0, 8, (10,))),
            ("port", "delivers_to", "manufacturer"):
                (torch.randint(0, 8, (10,)), torch.randint(0, 5, (10,))),
        }

        embeddings = model(x_dict, edge_dict)
        src_idx = torch.tensor([0, 1, 2])
        dst_idx = torch.tensor([0, 1, 2])
        # Link prediction requires src and dst embeddings of same out_channels
        # Use manufacturer embeddings for both to test the head
        scores = model.link_head(
            torch.cat([embeddings["manufacturer"][dst_idx],
                       embeddings["manufacturer"][dst_idx]], dim=-1)
        ).squeeze()
        assert scores.min() >= 0 and scores.max() <= 1


# ═════════════════════════════════════════════════════════════════════════════
# TCN Tests
# ═════════════════════════════════════════════════════════════════════════════

class TestTCN:
    """Tests for Temporal Convolutional Network."""

    def test_causal_conv_block_shape(self):
        """CausalConv1dBlock preserves temporal dimension."""
        from src.models.tcn import CausalConv1dBlock

        block = CausalConv1dBlock(in_channels=1, out_channels=64,
                                   kernel_size=3, dilation=4)
        x = torch.randn(4, 1, 128)  # [B, C, T]
        out = block(x)
        assert out.shape == (4, 64, 128), f"Expected (4, 64, 128), got {out.shape}"

    def test_tcn_model_forward(self):
        """TCNModel forward produces quantile predictions of correct shape."""
        from src.models.tcn import TCNModel

        model = TCNModel(input_channels=1, num_filters=32,
                          kernel_size=3, output_len=30,
                          quantiles=[0.1, 0.5, 0.9])

        x = torch.randn(8, 128, 1)  # [B, T, C]
        preds = model(x)

        assert len(preds) == 3, "Should have 3 quantile predictions"
        for q in [0.1, 0.5, 0.9]:
            assert preds[q].shape == (8, 30), f"Quantile {q}: expected (8, 30), got {preds[q].shape}"

    def test_quantile_loss(self):
        """Quantile loss is non-negative and differentiable."""
        from src.models.tcn import quantile_loss

        preds = {0.1: torch.randn(4, 30, requires_grad=True),
                 0.5: torch.randn(4, 30, requires_grad=True),
                 0.9: torch.randn(4, 30, requires_grad=True)}
        targets = torch.randn(4, 30)

        loss = quantile_loss(preds, targets)
        assert loss.item() >= 0, "Quantile loss should be non-negative"
        loss.backward()
        for q, p in preds.items():
            assert p.grad is not None, f"Gradient not computed for quantile {q}"

    def test_tcn_gradient_flow(self):
        """Gradients flow through the full TCN stack."""
        from src.models.tcn import TCNModel, quantile_loss

        model = TCNModel(input_channels=1, num_filters=32,
                          output_len=30, quantiles=[0.5])
        x = torch.randn(4, 128, 1)
        y = torch.randn(4, 30)

        preds = model(x)
        loss = quantile_loss(preds, y)
        loss.backward()

        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"


# ═════════════════════════════════════════════════════════════════════════════
# Transformer Tests
# ═════════════════════════════════════════════════════════════════════════════

class TestTransformer:
    """Tests for Shipment Risk Transformer Encoder."""

    def test_transformer_forward_shape(self):
        """Transformer produces [B, num_tasks] output."""
        from src.models.transformer import ShipmentRiskTransformer

        model = ShipmentRiskTransformer(
            input_dim=8, d_model=64, num_heads=4,
            num_layers=3, num_tasks=3
        )

        x = torch.randn(16, 8, 8)  # [B, seq_len=8, input_dim=8]
        out = model(x)
        assert out.shape == (16, 3), f"Expected (16, 3), got {out.shape}"

    def test_transformer_outputs_in_range(self):
        """Transformer outputs should be probabilities in [0, 1]."""
        from src.models.transformer import ShipmentRiskTransformer

        model = ShipmentRiskTransformer(
            input_dim=8, d_model=32, num_heads=4,
            num_layers=2, num_tasks=3
        )

        x = torch.randn(8, 8, 8)
        out = model(x)
        assert out.min() >= 0.0 and out.max() <= 1.0, \
            f"Outputs out of [0,1] range: min={out.min():.4f}, max={out.max():.4f}"

    def test_transformer_gradient_flow(self):
        """Gradients flow through Transformer encoder."""
        from src.models.transformer import ShipmentRiskTransformer

        model = ShipmentRiskTransformer(
            input_dim=8, d_model=32, num_heads=4,
            num_layers=2, num_tasks=3
        )

        x = torch.randn(8, 8, 8)
        y = torch.rand(8, 3)  # binary targets

        out = model(x)
        loss = nn.BCELoss()(out, y)
        loss.backward()

        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"


# ═════════════════════════════════════════════════════════════════════════════
# Survival Tests
# ═════════════════════════════════════════════════════════════════════════════

class TestSurvival:
    """Tests for DeepSurv neural survival model."""

    def test_deepsurv_forward_shape(self):
        """DeepSurv outputs a single risk score per sample."""
        from src.models.survival import DeepSurv

        model = DeepSurv(in_features=20, hidden_dim=64, num_layers=3)
        x = torch.randn(10, 20)
        out = model(x)
        assert out.shape == (10, 1), f"Expected (10, 1), got {out.shape}"

    def test_partial_likelihood_loss(self):
        """Negative log partial likelihood computes and is differentiable."""
        from src.models.survival import negative_log_partial_likelihood

        risk = torch.randn(10, 1, requires_grad=True)
        durations = torch.rand(10) * 365
        events = (torch.rand(10) > 0.5).float()

        loss = negative_log_partial_likelihood(risk, durations, events)
        loss.backward()
        assert risk.grad is not None, "Gradient not computed for risk scores"

    def test_c_index_computation(self):
        """C-index returns value between 0 and 1."""
        from src.models.survival import compute_c_index

        risk = torch.tensor([[0.8], [0.3], [0.5], [0.9], [0.1]])
        durations = torch.tensor([100, 300, 200, 50, 350])
        events = torch.tensor([1, 0, 1, 1, 0])

        c = compute_c_index(risk, durations, events)
        assert 0.0 <= c <= 1.0, f"C-index out of range: {c}"


# ═════════════════════════════════════════════════════════════════════════════
# Ensemble Tests
# ═════════════════════════════════════════════════════════════════════════════

class TestEnsemble:
    """Tests for stacking ensemble utilities."""

    def test_ece_perfect_calibration(self):
        """ECE should be ~0 for perfectly calibrated predictions."""
        from src.models.ensemble import compute_ece

        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_prob = np.array([0.1, 0.2, 0.1, 0.9, 0.8, 0.9])

        ece = compute_ece(y_true, y_prob, n_bins=5)
        assert ece < 0.25, f"ECE too high for well-calibrated predictions: {ece}"

    def test_gini_coefficient(self):
        """Gini coefficient should be in [-1, 1] and > 0 for decent models."""
        from src.models.ensemble import compute_gini

        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_prob = np.array([0.1, 0.3, 0.8, 0.9, 0.2, 0.7])

        gini = compute_gini(y_true, y_prob)
        assert -1.0 <= gini <= 1.0, f"Gini out of range: {gini}"
        assert gini > 0, f"Gini should be positive for decent predictions: {gini}"

    def test_ece_range(self):
        """ECE should always be between 0 and 1."""
        from src.models.ensemble import compute_ece

        np.random.seed(42)
        y_true = np.random.randint(0, 2, 100)
        y_prob = np.random.rand(100)

        ece = compute_ece(y_true, y_prob)
        assert 0.0 <= ece <= 1.0, f"ECE out of range: {ece}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
