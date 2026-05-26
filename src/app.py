import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import pickle
import sys

# Ensure UTF-8 output on Windows stdout
sys.stdout.reconfigure(encoding='utf-8')

# Add project root to sys.path so 'from src...' imports work with streamlit
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# Custom CSS for dark-mode glassmorphism premium styling
st.set_page_config(
    page_title="LogisChain AI - Predictive Trade Finance & Logistics Valuation",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium theme styles
st.markdown("""
<style>
    /* Dark Mode Glassmorphism Theme */
    .stApp {
        background: radial-gradient(circle at top right, #0d1117, #010409);
        color: #c9d1d9;
        font-family: 'Outfit', 'Inter', sans-serif;
    }
    
    /* Header & Title styles */
    h1, h2, h3 {
        color: #ffffff !important;
        font-weight: 700;
        letter-spacing: -0.02em;
    }
    
    /* Premium Glassmorphic Cards */
    .glass-card {
        background: rgba(22, 27, 34, 0.4);
        border: 1px solid rgba(240, 246, 252, 0.1);
        border-radius: 12px;
        padding: 24px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        margin-bottom: 20px;
        transition: transform 0.2s, border-color 0.2s;
    }
    
    .glass-card:hover {
        transform: translateY(-2px);
        border-color: rgba(56, 139, 253, 0.4);
    }
    
    /* Metrics panel */
    .metric-value {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(120deg, #58a6ff, #bc8cff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# 1. State Management Setup
# ─────────────────────────────────────────────────────────────────────────────

from src.simulation.engine import SimulationEngine
from src.simulation.scenarios import get_default_scenarios
from src.simulation.modes import run_portfolio_management_mode, run_scf_pricing_mode

if "engine" not in st.session_state:
    engine = SimulationEngine()
    engine.load_scenarios(get_default_scenarios())
    st.session_state.engine = engine
    st.session_state.history = [engine.get_state_snapshot()]
    st.session_state.game_mode = "Portfolio Management"
    st.session_state.decisions_queue = []

engine = st.session_state.engine
snapshot = st.session_state.history[-1]

# ─────────────────────────────────────────────────────────────────────────────
# 2. Sidebar Navigation
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.image("https://img.icons8.com/nolan/96/blockchain.png", width=70)
    st.title("LogisChain Lab")
    st.markdown("---")
    
    # Game Mode selection
    mode = st.radio("Select Game Mode", ["Portfolio Management", "SCF dynamic pricing"], key="mode_selector")
    if mode != st.session_state.game_mode:
        st.session_state.game_mode = mode
        # Re-initialize engine
        engine = SimulationEngine()
        engine.load_scenarios(get_default_scenarios())
        if mode == "Portfolio Management":
            run_portfolio_management_mode(engine)
        else:
            run_scf_pricing_mode(engine)
        st.session_state.engine = engine
        st.session_state.history = [engine.get_state_snapshot()]
        st.session_state.decisions_queue = []
        st.rerun()
        
    st.markdown("---")
    st.subheader("Controls")
    
    # Turn advancement
    if st.button("Advance simulated week 🚀", type="primary", use_container_width=True):
        snap = engine.advance_turn(st.session_state.decisions_queue)
        st.session_state.history.append(snap)
        st.session_state.decisions_queue = []
        st.rerun()
        
    if st.button("Reset Simulation 🔄", use_container_width=True):
        engine = SimulationEngine()
        engine.load_scenarios(get_default_scenarios())
        if st.session_state.game_mode == "Portfolio Management":
            run_portfolio_management_mode(engine)
        else:
            run_scf_pricing_mode(engine)
        st.session_state.engine = engine
        st.session_state.history = [engine.get_state_snapshot()]
        st.session_state.decisions_queue = []
        st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# 3. Main Dashboard Layout
# ─────────────────────────────────────────────────────────────────────────────

st.title("🌌 LogisChain AI Control Center")
st.markdown("Predictive Trade Finance & Multi-Echelon Supply Chain Risk Optimization Engine.")

# Top metrics bar
cols = st.columns(4)
with cols[0]:
    st.markdown(f"""
    <div class="glass-card">
        <div class="metric-label">Simulated Week</div>
        <div class="metric-value">W{snapshot['week']} / W52</div>
    </div>
    """, unsafe_allow_html=True)
with cols[1]:
    st.markdown(f"""
    <div class="glass-card">
        <div class="metric-label">Cash Position</div>
        <div class="metric-value">${snapshot['player']['cash']:,.2f}</div>
    </div>
    """, unsafe_allow_html=True)
with cols[2]:
    st.markdown(f"""
    <div class="glass-card">
        <div class="metric-label">Total Portfolio Value</div>
        <div class="metric-value">${snapshot['player']['portfolio_value']:,.2f}</div>
    </div>
    """, unsafe_allow_html=True)
with cols[3]:
    st.markdown(f"""
    <div class="glass-card">
        <div class="metric-label">LogisChain Score</div>
        <div class="metric-value">{snapshot['player']['score']:.1f} pts</div>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Tab 1: Real-Time Operational Controls
# ─────────────────────────────────────────────────────────────────────────────
tabs = st.tabs(["🎮 Gamified Simulation Lab", "📈 Intelligence & Predictive Modeling", "🗺️ Dynamic Supply Chain Network"])

with tabs[0]:
    st.subheader("Interactive Operational Decisions Queue")
    
    d_cols = st.columns(2)
    with d_cols[0]:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.write("### 📜 Issue Letter of Credit (LC)")
        
        # Pull suppliers
        suppliers = [n for n in engine.nodes.values() if n.node_type == "supplier"]
        sup_names = [s.name for s in suppliers]
        selected_beneficiary = st.selectbox("Beneficiary (Supplier)", sup_names)
        lc_amount = st.number_input("LC Amount (USD)", min_value=10_000, value=250_000, step=10_000)
        validity = st.slider("Validity (Weeks)", 4, 24, 12)
        
        if st.button("Add LC to decisions queue"):
            st.session_state.decisions_queue.append({
                "action": "issue_lc",
                "beneficiary": selected_beneficiary,
                "applicant": "Player Corp",
                "amount": float(lc_amount),
                "validity_turns": validity
            })
            st.success("LC added!")
        st.markdown('</div>', unsafe_allow_html=True)

    with d_cols[1]:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.write("### 🚢 Create Shipment")
        
        # Link origins/destinations
        origin = st.selectbox("Origin Node", [n.node_id for n in engine.nodes.values() if n.node_type == "supplier"])
        dest = st.selectbox("Destination Node", [n.node_id for n in engine.nodes.values() if n.node_type == "port"])
        ship_val = st.number_input("Shipment Value (USD)", min_value=10_000, value=200_000, step=10_000)
        insure = st.checkbox("Purchase 2% Cargo Insurance", value=True)
        
        # Match LCs
        active_lcs = [lc.lc_id for lc in engine.player.letters_of_credit if lc.status == "active"]
        selected_lc = st.selectbox("Attach LC (Optional)", ["None"] + active_lcs)
        
        if st.button("Add Shipment to decisions queue"):
            st.session_state.decisions_queue.append({
                "action": "create_shipment",
                "origin_id": origin,
                "dest_id": dest,
                "commodity": "Industrial Ratios",
                "value": float(ship_val),
                "insure": insure,
                "lc_id": None if selected_lc == "None" else selected_lc
            })
            st.success("Shipment added!")
        st.markdown('</div>', unsafe_allow_html=True)

    # Show decisions queue
    st.write("#### Decisions Queued for this Simulated Turn:")
    if st.session_state.decisions_queue:
        st.write(st.session_state.decisions_queue)
    else:
        st.info("No decisions queued yet. Select actions above and click 'Add' to queue them.")

    # Scenarios active panel
    st.write("### 📢 Active Environmental / Shock Events")
    if snapshot["active_scenarios"]:
        for sc in snapshot["active_scenarios"]:
            st.warning(f"**{sc['name']}** (Severity: {sc['severity']:.2f}) - {sc['type']}")
    else:
        st.success("Clear weather! No active environmental disruption events.")

with tabs[1]:
    st.subheader("LogisChain Predictive Models Integration")
    
    # Load Model artifacts for display
    st.write("#### 🛡️ Survival Probability of Default (SC-PD Model)")
    st.write("Incorporates multi-echelon spatial graph centralities & port congestion signals with financial ratios.")
    
    if os.path.exists("data/models/model_card_scpd.md"):
        with st.expander("📄 View SC-PD Model Card"):
            with open("data/models/model_card_scpd.md", "r", encoding="utf-8") as f:
                st.markdown(f.read())

    # Live Predictor Widget using Synthetic input
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.write("### 🛠️ Interactive Credit Risk & Covenant Stress Test")
    
    col_pred = st.columns(3)
    with col_pred[0]:
        test_cur = st.slider("Current Ratio", 0.5, 3.5, 1.5)
        test_de = st.slider("Debt-to-Equity Ratio", 0.1, 5.0, 1.2)
    with col_pred[1]:
        test_otif = st.slider("OTIF Rate", 0.5, 1.0, 0.95)
        test_lead = st.slider("Mean Lead Time (Days)", 3.0, 30.0, 12.0)
    with col_pred[2]:
        test_ccc = st.slider("Current CCC (Days)", 10.0, 150.0, 60.0)
        
    # Quick live warning
    if test_ccc > 75.0:
        st.error(f"🚨 ALERT: Cash Conversion Cycle ({test_ccc:.1f} days) exceeds 75-day risk covenant limit!")
    else:
        st.success(f"Covenant Status: Active (CCC {test_ccc:.1f} days is within safety thresholds)")
    st.markdown('</div>', unsafe_allow_html=True)

with tabs[2]:
    st.subheader("Dynamic Supply Chain Topology & Logistics Nodes")
    st.write("Physical Layer simulation mapping 100+ nodes and 500+ transportation corridors.")
    
    # Generate interactive node type counts
    node_sum = engine.get_network_summary()
    st.write(node_sum)
    
    # Draw simple Plotly distribution of nodes by region/modality
    mod_df = pd.DataFrame({
        "Modality": list(node_sum["modality_counts"].keys()),
        "Corridors": list(node_sum["modality_counts"].values())
    })
    fig = px.bar(mod_df, x="Modality", y="Corridors", color="Modality", title="Transportation Modality Distribution", template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)
