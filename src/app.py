import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import pickle
import sys

# Ensure UTF-8 output on Windows stdout
sys.stdout.reconfigure(encoding='utf-8')

# Add project root to sys.path so 'from src...' imports work with streamlit
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# ─────────────────────────────────────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="LogisChain AI - Predictive Trade Finance & Logistics Valuation",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────────────────────
# Premium CSS Theme
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    /* ── Google Fonts ──────────────────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    /* ── Animated Background ──────────────────────────────── */
    @keyframes gradientShift {
        0%   { background-position: 0% 50%; }
        50%  { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    @keyframes pulseGlow {
        0%, 100% { box-shadow: 0 0 20px rgba(88, 166, 255, 0.08); }
        50%      { box-shadow: 0 0 30px rgba(88, 166, 255, 0.18); }
    }

    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(12px); }
        to   { opacity: 1; transform: translateY(0); }
    }

    @keyframes borderGlow {
        0%, 100% { border-color: rgba(88, 166, 255, 0.15); }
        50%      { border-color: rgba(188, 140, 255, 0.30); }
    }

    /* ── Root Variables ───────────────────────────────────── */
    :root {
        --bg-primary: #0d1117;
        --bg-secondary: #161b22;
        --bg-card: rgba(22, 27, 34, 0.55);
        --border-subtle: rgba(240, 246, 252, 0.08);
        --border-accent: rgba(56, 139, 253, 0.35);
        --text-primary: #e6edf3;
        --text-secondary: #8b949e;
        --text-muted: #6e7681;
        --accent-blue: #58a6ff;
        --accent-purple: #bc8cff;
        --accent-green: #3fb950;
        --accent-amber: #d29922;
        --accent-red: #f85149;
        --accent-cyan: #39d353;
        --font-main: 'Outfit', 'Inter', -apple-system, sans-serif;
        --font-mono: 'JetBrains Mono', 'Fira Code', monospace;
    }

    /* ── App Shell ────────────────────────────────────────── */
    .stApp {
        background: linear-gradient(135deg, #0d1117 0%, #0a0e14 35%, #0d1117 65%, #101820 100%);
        background-size: 200% 200%;
        animation: gradientShift 20s ease infinite;
        color: var(--text-primary);
        font-family: var(--font-main);
    }

    /* ── Typography ───────────────────────────────────────── */
    h1 {
        font-family: var(--font-main) !important;
        font-weight: 800 !important;
        letter-spacing: -0.03em !important;
        background: linear-gradient(135deg, #58a6ff 0%, #bc8cff 50%, #f778ba 100%);
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        background-clip: text !important;
    }

    h2, h3 {
        font-family: var(--font-main) !important;
        color: #e6edf3 !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em !important;
    }

    p, label, .stMarkdown {
        font-family: var(--font-main) !important;
    }

    /* ── Sidebar ──────────────────────────────────────────── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(13, 17, 23, 0.95) 0%, rgba(22, 27, 34, 0.92) 100%) !important;
        border-right: 1px solid rgba(88, 166, 255, 0.12) !important;
        backdrop-filter: blur(20px) !important;
    }

    section[data-testid="stSidebar"] .stMarkdown h1 {
        font-size: 1.4rem !important;
        background: linear-gradient(120deg, #58a6ff, #bc8cff) !important;
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
    }

    /* ── Glassmorphic Cards ───────────────────────────────── */
    .glass-card {
        background: var(--bg-card);
        border: 1px solid var(--border-subtle);
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.25), inset 0 1px 0 rgba(255,255,255,0.04);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        margin-bottom: 16px;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        animation: fadeInUp 0.5s ease-out, borderGlow 4s ease-in-out infinite;
    }

    .glass-card:hover {
        transform: translateY(-3px);
        border-color: var(--border-accent);
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.35), 0 0 20px rgba(88, 166, 255, 0.08);
    }

    /* ── Metric Cards ─────────────────────────────────────── */
    .metric-card {
        background: linear-gradient(135deg, rgba(22, 27, 34, 0.6) 0%, rgba(13, 17, 23, 0.7) 100%);
        border: 1px solid var(--border-subtle);
        border-radius: 16px;
        padding: 20px 24px;
        box-shadow: 0 4px 24px rgba(0, 0, 0, 0.2);
        backdrop-filter: blur(16px);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        animation: fadeInUp 0.4s ease-out;
        position: relative;
        overflow: hidden;
    }

    .metric-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        background: linear-gradient(90deg, var(--accent-blue), var(--accent-purple));
        border-radius: 16px 16px 0 0;
        opacity: 0;
        transition: opacity 0.3s ease;
    }

    .metric-card:hover::before { opacity: 1; }

    .metric-card:hover {
        transform: translateY(-2px);
        border-color: var(--border-accent);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3), 0 0 15px rgba(88, 166, 255, 0.06);
    }

    .metric-icon {
        font-size: 1.6rem;
        margin-bottom: 4px;
        display: block;
    }

    .metric-label {
        font-size: 0.72rem;
        font-weight: 600;
        color: var(--text-secondary);
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 6px;
    }

    .metric-value {
        font-size: 1.7rem;
        font-weight: 800;
        background: linear-gradient(120deg, #58a6ff, #bc8cff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        line-height: 1.1;
    }

    .metric-delta {
        font-size: 0.78rem;
        font-weight: 600;
        margin-top: 6px;
        font-family: var(--font-mono);
    }

    .delta-up   { color: var(--accent-green); }
    .delta-down { color: var(--accent-red); }
    .delta-flat { color: var(--text-muted); }

    /* ── Progress Bar ─────────────────────────────────────── */
    .sim-progress-outer {
        background: rgba(255,255,255,0.05);
        border-radius: 8px;
        height: 10px;
        overflow: hidden;
        margin: 8px 0 16px 0;
    }

    .sim-progress-inner {
        height: 100%;
        border-radius: 8px;
        background: linear-gradient(90deg, #58a6ff, #bc8cff, #f778ba);
        background-size: 200% 100%;
        animation: gradientShift 3s ease infinite;
        transition: width 0.6s cubic-bezier(0.4, 0, 0.2, 1);
    }

    /* ── Stat Badges ──────────────────────────────────────── */
    .sidebar-stat {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 8px 12px;
        border-radius: 10px;
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.05);
        margin-bottom: 6px;
        font-size: 0.82rem;
        transition: background 0.2s;
    }

    .sidebar-stat:hover { background: rgba(255,255,255,0.06); }

    .stat-label { color: var(--text-secondary); font-weight: 500; }

    .stat-value {
        color: var(--text-primary);
        font-weight: 700;
        font-family: var(--font-mono);
    }

    .stat-value-accent {
        color: var(--accent-blue);
        font-weight: 700;
        font-family: var(--font-mono);
    }

    /* ── Status Pills ─────────────────────────────────────── */
    .status-pill {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 999px;
        font-size: 0.72rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }

    .pill-active   { background: rgba(63, 185, 80, 0.15); color: #3fb950; border: 1px solid rgba(63, 185, 80, 0.3); }
    .pill-transit  { background: rgba(88, 166, 255, 0.15); color: #58a6ff; border: 1px solid rgba(88, 166, 255, 0.3); }
    .pill-expired  { background: rgba(139, 148, 158, 0.15); color: #8b949e; border: 1px solid rgba(139, 148, 158, 0.3); }
    .pill-damaged  { background: rgba(248, 81, 73, 0.15); color: #f85149; border: 1px solid rgba(248, 81, 73, 0.3); }
    .pill-delivered { background: rgba(57, 211, 83, 0.15); color: #39d353; border: 1px solid rgba(57, 211, 83, 0.3); }
    .pill-warning  { background: rgba(210, 153, 34, 0.15); color: #d29922; border: 1px solid rgba(210, 153, 34, 0.3); }

    /* ── Decision Card ────────────────────────────────────── */
    .decision-item {
        background: rgba(88, 166, 255, 0.06);
        border: 1px solid rgba(88, 166, 255, 0.15);
        border-radius: 12px;
        padding: 14px 18px;
        margin-bottom: 8px;
        display: flex;
        align-items: center;
        gap: 12px;
        font-size: 0.88rem;
        transition: all 0.2s;
    }

    .decision-item:hover {
        background: rgba(88, 166, 255, 0.10);
        border-color: rgba(88, 166, 255, 0.25);
    }

    .decision-icon { font-size: 1.3rem; }
    .decision-text { color: var(--text-primary); flex: 1; }
    .decision-amount { color: var(--accent-blue); font-weight: 700; font-family: var(--font-mono); }

    /* ── Scenario Alert Card ──────────────────────────────── */
    .scenario-alert {
        background: linear-gradient(135deg, rgba(210, 153, 34, 0.08) 0%, rgba(248, 81, 73, 0.06) 100%);
        border: 1px solid rgba(210, 153, 34, 0.25);
        border-left: 4px solid var(--accent-amber);
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 10px;
        animation: fadeInUp 0.4s ease-out;
    }

    .scenario-name {
        font-weight: 700;
        color: var(--accent-amber);
        font-size: 0.95rem;
        margin-bottom: 4px;
    }

    .scenario-detail {
        color: var(--text-secondary);
        font-size: 0.82rem;
    }

    .scenario-severity {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 6px;
        font-size: 0.72rem;
        font-weight: 700;
        font-family: var(--font-mono);
        background: rgba(248, 81, 73, 0.15);
        color: #f85149;
        margin-left: 8px;
    }

    /* ── Timeline ─────────────────────────────────────────── */
    .timeline-bar {
        position: relative;
        background: rgba(255,255,255,0.05);
        border-radius: 8px;
        height: 36px;
        margin: 10px 0 20px 0;
        overflow: visible;
    }

    .timeline-marker {
        position: absolute;
        top: -4px;
        width: 14px; height: 14px;
        border-radius: 50%;
        border: 2px solid var(--accent-amber);
        background: var(--bg-primary);
        z-index: 2;
        transition: all 0.3s;
    }

    .timeline-marker:hover {
        transform: scale(1.4);
        box-shadow: 0 0 12px rgba(210, 153, 34, 0.4);
    }

    .timeline-now {
        position: absolute;
        top: 0; bottom: 0;
        background: linear-gradient(90deg, rgba(88, 166, 255, 0.15), transparent);
        border-radius: 8px 0 0 8px;
        border-right: 2px solid var(--accent-blue);
    }

    /* ── Tabs ──────────────────────────────────────────────── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background: rgba(22, 27, 34, 0.3);
        border-radius: 12px;
        padding: 4px;
        border: 1px solid var(--border-subtle);
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        font-weight: 600;
        font-size: 0.85rem;
        color: var(--text-secondary);
        transition: all 0.2s;
    }

    .stTabs [aria-selected="true"] {
        background: rgba(88, 166, 255, 0.12) !important;
        color: var(--accent-blue) !important;
    }

    /* ── Buttons ───────────────────────────────────────────── */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #58a6ff, #bc8cff) !important;
        border: none !important;
        font-weight: 700 !important;
        letter-spacing: 0.02em !important;
        border-radius: 10px !important;
        transition: all 0.3s !important;
    }

    .stButton > button[kind="primary"]:hover {
        box-shadow: 0 4px 20px rgba(88, 166, 255, 0.3) !important;
        transform: translateY(-1px) !important;
    }

    /* ── Footer ────────────────────────────────────────────── */
    .app-footer {
        margin-top: 60px;
        padding: 20px 0;
        text-align: center;
        border-top: 1px solid var(--border-subtle);
        color: var(--text-muted);
        font-size: 0.78rem;
        letter-spacing: 0.02em;
    }

    .footer-brand {
        background: linear-gradient(120deg, #58a6ff, #bc8cff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
    }

    /* ── Plotly chart containers ───────────────────────────── */
    .stPlotlyChart {
        border-radius: 12px;
        overflow: hidden;
    }

    /* ── Hide Streamlit default footer ─────────────────────── */
    footer { visibility: hidden; }
    #MainMenu { visibility: hidden; }

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
prev_snapshot = st.session_state.history[-2] if len(st.session_state.history) > 1 else snapshot


# ─────────────────────────────────────────────────────────────────────────────
# Helper: delta formatting
# ─────────────────────────────────────────────────────────────────────────────

def fmt_delta(current, previous, prefix="$", suffix="", is_pct=False):
    """Return HTML span for a delta value."""
    diff = current - previous
    if abs(diff) < 0.01:
        return '<span class="metric-delta delta-flat">― no change</span>'
    sign = "▲" if diff > 0 else "▼"
    css = "delta-up" if diff > 0 else "delta-down"
    if is_pct:
        return f'<span class="metric-delta {css}">{sign} {abs(diff):.1f}%{suffix}</span>'
    if abs(diff) >= 1_000_000:
        return f'<span class="metric-delta {css}">{sign} {prefix}{abs(diff)/1_000_000:,.2f}M{suffix}</span>'
    elif abs(diff) >= 1_000:
        return f'<span class="metric-delta {css}">{sign} {prefix}{abs(diff)/1_000:,.1f}K{suffix}</span>'
    else:
        return f'<span class="metric-delta {css}">{sign} {prefix}{abs(diff):,.2f}{suffix}</span>'


def get_status_pill(status):
    """Return HTML pill for a status string."""
    pill_map = {
        "active": ("pill-active", "● ACTIVE"),
        "in_transit": ("pill-transit", "● IN TRANSIT"),
        "delivered": ("pill-delivered", "✓ DELIVERED"),
        "damaged": ("pill-damaged", "✕ DAMAGED"),
        "expired": ("pill-expired", "○ EXPIRED"),
        "delayed": ("pill-warning", "⏱ DELAYED"),
        "drawn": ("pill-active", "● DRAWN"),
        "cancelled": ("pill-expired", "○ CANCELLED"),
    }
    css, label = pill_map.get(status, ("pill-expired", status.upper()))
    return f'<span class="status-pill {css}">{label}</span>'


# ─────────────────────────────────────────────────────────────────────────────
# 2. Sidebar
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("# ⛓️ LogisChain Lab")
    st.caption("Predictive Trade Finance Simulator")
    st.markdown("---")

    # Game Mode selection
    mode = st.radio(
        "Select Game Mode",
        ["Portfolio Management", "SCF dynamic pricing"],
        key="mode_selector",
        help="Portfolio: Manage LCs & shipments for max returns. SCF: Dynamic pricing as a supply chain funder."
    )
    if mode != st.session_state.game_mode:
        st.session_state.game_mode = mode
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

    # Simulation progress bar
    week = snapshot['week']
    progress_pct = min(week / 52 * 100, 100)
    st.markdown(f"""
    <div style="font-size:0.78rem; color: var(--text-secondary); font-weight:600; margin-bottom:2px;">
        SIMULATION PROGRESS — Week {week} of 52
    </div>
    <div class="sim-progress-outer">
        <div class="sim-progress-inner" style="width:{progress_pct}%"></div>
    </div>
    """, unsafe_allow_html=True)

    # Quick stats
    st.markdown("**Live Stats**")
    net = snapshot.get('network', {})
    st.markdown(f"""
    <div class="sidebar-stat">
        <span class="stat-label">🚢 Active Shipments</span>
        <span class="stat-value">{snapshot['player']['active_shipments']}</span>
    </div>
    <div class="sidebar-stat">
        <span class="stat-label">📜 Active LCs</span>
        <span class="stat-value">{snapshot['player']['active_lcs']}</span>
    </div>
    <div class="sidebar-stat">
        <span class="stat-label">✅ Delivered</span>
        <span class="stat-value">{snapshot['player']['completed_shipments']}</span>
    </div>
    <div class="sidebar-stat">
        <span class="stat-label">🌐 Network Nodes</span>
        <span class="stat-value">{net.get('operational_nodes', 0)}/{net.get('total_nodes', 0)}</span>
    </div>
    <div class="sidebar-stat">
        <span class="stat-label">⚠️ Disrupted Links</span>
        <span class="stat-value-accent">{net.get('disrupted_links', 0)}</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Player vs AI mini comparison
    st.markdown("**Player vs AI Opponent**")
    p_score = snapshot['player']['score']
    ai_score = snapshot['ai_opponent']['score']
    max_s = max(p_score, ai_score, 1)
    st.markdown(f"""
    <div style="margin-bottom:8px;">
        <div style="font-size:0.78rem; color: var(--accent-blue); font-weight:600;">👤 You — {p_score:.0f} pts</div>
        <div class="sim-progress-outer" style="height:6px; margin:4px 0 8px 0;">
            <div style="height:100%;width:{p_score/max_s*100:.0f}%;border-radius:6px;background:linear-gradient(90deg,#58a6ff,#bc8cff);"></div>
        </div>
        <div style="font-size:0.78rem; color: var(--accent-amber); font-weight:600;">🤖 AI — {ai_score:.0f} pts</div>
        <div class="sim-progress-outer" style="height:6px; margin:4px 0 0 0;">
            <div style="height:100%;width:{ai_score/max_s*100:.0f}%;border-radius:6px;background:linear-gradient(90deg,#d29922,#f85149);"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**Controls**")

    # Turn advancement
    if st.button("⚡ Advance Simulated Week", type="primary", use_container_width=True):
        snap = engine.advance_turn(st.session_state.decisions_queue)
        st.session_state.history.append(snap)
        st.session_state.decisions_queue = []
        st.rerun()

    if st.button("🔄 Reset Simulation", use_container_width=True):
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
# 3. Main Header
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("# 🌌 LogisChain AI Control Center")
st.markdown(
    '<span style="color:#8b949e; font-size:1rem;">Predictive Trade Finance & Multi-Echelon Supply Chain Risk Optimization Engine</span>',
    unsafe_allow_html=True
)
st.markdown("")

# ─────────────────────────────────────────────────────────────────────────────
# 4. Top Metrics Bar (6 cards)
# ─────────────────────────────────────────────────────────────────────────────

net_info = snapshot.get('network', {})
operational_pct = (net_info.get('operational_nodes', 0) / max(net_info.get('total_nodes', 1), 1)) * 100

cols = st.columns(6)

metric_data = [
    ("📅", "Simulated Week", f"W{snapshot['week']} / W52", None),
    ("💰", "Cash Position", f"${snapshot['player']['cash']:,.0f}",
     fmt_delta(snapshot['player']['cash'], prev_snapshot['player']['cash'])),
    ("📊", "Portfolio Value", f"${snapshot['player']['portfolio_value']:,.0f}",
     fmt_delta(snapshot['player']['portfolio_value'], prev_snapshot['player']['portfolio_value'])),
    ("🏆", "LogisChain Score", f"{snapshot['player']['score']:.0f} pts",
     fmt_delta(snapshot['player']['score'], prev_snapshot['player']['score'], prefix="")),
    ("🚢", "Active Shipments", f"{snapshot['player']['active_shipments']}",
     fmt_delta(snapshot['player']['active_shipments'], prev_snapshot['player']['active_shipments'], prefix="")),
    ("🌐", "Network Health", f"{operational_pct:.0f}%",
     None),
]

for col, (icon, label, value, delta) in zip(cols, metric_data):
    delta_html = delta if delta else ""
    with col:
        st.markdown(f"""
        <div class="metric-card">
            <span class="metric-icon">{icon}</span>
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            {delta_html}
        </div>
        """, unsafe_allow_html=True)

st.markdown("")

# ─────────────────────────────────────────────────────────────────────────────
# 5. Tabs
# ─────────────────────────────────────────────────────────────────────────────

tabs = st.tabs([
    "🎮 Gamified Simulation Lab",
    "📈 Intelligence & Predictive Modeling",
    "🗺️ Dynamic Supply Chain Network"
])

# ═════════════════════════════════════════════════════════════════════════════
# TAB 1: GAMIFIED SIMULATION LAB
# ═════════════════════════════════════════════════════════════════════════════

with tabs[0]:
    st.markdown("### 🎯 Interactive Operational Decisions")
    st.markdown("")

    d_cols = st.columns(2)

    # ── Issue LC Card ─────────────────────────────────────────────────────────
    with d_cols[0]:
        with st.container(border=True):
            st.markdown("#### 📜 Issue Letter of Credit")

            suppliers = [n for n in engine.nodes.values() if n.node_type == "supplier"]
            sup_names = [s.name for s in suppliers]
            selected_beneficiary = st.selectbox("Beneficiary (Supplier)", sup_names, key="lc_beneficiary")
            lc_amount = st.number_input("LC Amount (USD)", min_value=10_000, value=250_000, step=10_000, key="lc_amount")
            validity = st.slider("Validity (Weeks)", 4, 24, 12, key="lc_validity")

            fee_preview = lc_amount * 0.015
            st.markdown(f'<span style="font-size:0.8rem; color:var(--text-secondary);">Issuance fee: <b style="color:var(--accent-amber);">${fee_preview:,.0f}</b> (1.5%)</span>', unsafe_allow_html=True)

            if st.button("➕ Add LC to Queue", key="btn_add_lc", use_container_width=True):
                st.session_state.decisions_queue.append({
                    "action": "issue_lc",
                    "beneficiary": selected_beneficiary,
                    "applicant": "Player Corp",
                    "amount": float(lc_amount),
                    "validity_turns": validity
                })
                st.success("LC queued!")

    # ── Create Shipment Card ──────────────────────────────────────────────────
    with d_cols[1]:
        with st.container(border=True):
            st.markdown("#### 🚢 Create Shipment")

            origin = st.selectbox("Origin Node", [n.node_id for n in engine.nodes.values() if n.node_type == "supplier"], key="ship_origin")
            dest = st.selectbox("Destination Node", [n.node_id for n in engine.nodes.values() if n.node_type == "port"], key="ship_dest")
            ship_val = st.number_input("Shipment Value (USD)", min_value=10_000, value=200_000, step=10_000, key="ship_value")
            insure = st.checkbox("Purchase 2% Cargo Insurance", value=True, key="ship_insure")

            active_lcs = [lc.lc_id for lc in engine.player.letters_of_credit if lc.status == "active"]
            selected_lc = st.selectbox("Attach LC (Optional)", ["None"] + active_lcs, key="ship_lc")

            cost_preview = ship_val * 0.03 + (ship_val * 0.02 if insure else 0)
            st.markdown(f'<span style="font-size:0.8rem; color:var(--text-secondary);">Est. cost: <b style="color:var(--accent-amber);">${cost_preview:,.0f}</b> (ship 3%{" + ins 2%" if insure else ""})</span>', unsafe_allow_html=True)

            if st.button("➕ Add Shipment to Queue", key="btn_add_ship", use_container_width=True):
                st.session_state.decisions_queue.append({
                    "action": "create_shipment",
                    "origin_id": origin,
                    "dest_id": dest,
                    "commodity": "Industrial Goods",
                    "value": float(ship_val),
                    "insure": insure,
                    "lc_id": None if selected_lc == "None" else selected_lc
                })
                st.success("Shipment queued!")

    # ── Decisions Queue ───────────────────────────────────────────────────────
    st.markdown("#### 📋 Decisions Queued for This Turn")
    if st.session_state.decisions_queue:
        for i, dec in enumerate(st.session_state.decisions_queue):
            action = dec.get("action", "")
            if action == "issue_lc":
                icon, text = "📜", f"Issue LC → <b>{dec.get('beneficiary', '?')}</b>"
                amt = f"${dec.get('amount', 0):,.0f}"
            elif action == "create_shipment":
                icon, text = "🚢", f"Shipment {dec.get('origin_id', '?')} → {dec.get('dest_id', '?')}"
                amt = f"${dec.get('value', 0):,.0f}"
            else:
                icon, text, amt = "⚙️", action, ""
            st.markdown(f"""
            <div class="decision-item">
                <span class="decision-icon">{icon}</span>
                <span class="decision-text">{text}</span>
                <span class="decision-amount">{amt}</span>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No decisions queued yet. Add LCs or shipments above, then advance the week.")

    st.markdown("---")

    # ── Active LCs Table ──────────────────────────────────────────────────────
    lcol, scol = st.columns(2)

    with lcol:
        st.markdown("#### 📜 Your Letters of Credit")
        if engine.player.letters_of_credit:
            lc_rows = []
            for lc in engine.player.letters_of_credit:
                lc_rows.append({
                    "LC ID": lc.lc_id,
                    "Beneficiary": lc.beneficiary,
                    "Amount": f"${lc.amount_usd:,.0f}",
                    "Issued": f"W{lc.issue_date}",
                    "Expires": f"W{lc.expiry_date}",
                    "Status": lc.status.title(),
                })
            st.dataframe(pd.DataFrame(lc_rows), use_container_width=True, hide_index=True)
        else:
            st.caption("No letters of credit issued yet.")

    # ── Active Shipments Table ────────────────────────────────────────────────
    with scol:
        st.markdown("#### 🚢 Active Shipments")
        if engine.player.active_shipments:
            sh_rows = []
            for s in engine.player.active_shipments:
                sh_rows.append({
                    "ID": s.shipment_id,
                    "Route": f"{s.origin_id} → {s.dest_id}",
                    "Value": f"${s.value_usd:,.0f}",
                    "ETA": f"W{s.expected_arrival_turn}",
                    "Delay": f"+{s.delay_days}d" if s.delay_days > 0 else "—",
                    "Insured": "✓" if s.insured else "✕",
                    "Status": s.status.replace("_", " ").title(),
                })
            st.dataframe(pd.DataFrame(sh_rows), use_container_width=True, hide_index=True)
        else:
            st.caption("No active shipments in transit.")

    st.markdown("---")

    # ── Scenario Timeline ─────────────────────────────────────────────────────
    st.markdown("#### 🗓️ Scenario Shock Timeline")
    scenarios_info = [
        (10, "Suez Canal Blockage", 0.85),
        (20, "Carrier Bankruptcy", 0.90),
        (30, "Port Workers Strike", 0.80),
        (40, "Demand Shock", 0.75),
        (48, "Tariff Escalation", 0.70),
    ]

    # Build a Plotly timeline
    fig_tl = go.Figure()

    # Current week marker
    fig_tl.add_trace(go.Scatter(
        x=[week], y=[0.5],
        mode='markers+text',
        marker=dict(size=16, color='#58a6ff', symbol='diamond', line=dict(width=2, color='#ffffff')),
        text=[f"W{week}"], textposition="top center",
        textfont=dict(color='#58a6ff', size=12, family='Outfit'),
        name="Current Week", showlegend=False
    ))

    # Scenario markers
    for turn, name, severity in scenarios_info:
        color = '#3fb950' if week > turn + 8 else ('#f85149' if week >= turn else '#d29922')
        fig_tl.add_trace(go.Scatter(
            x=[turn], y=[0.5],
            mode='markers+text',
            marker=dict(size=12 + severity * 8, color=color, symbol='circle',
                        line=dict(width=1.5, color='rgba(255,255,255,0.3)')),
            text=[name], textposition="top center",
            textfont=dict(color='#e6edf3', size=10, family='Outfit'),
            name=name, showlegend=False,
            hovertemplate=f"<b>{name}</b><br>Week {turn}<br>Severity: {severity:.0%}<extra></extra>"
        ))

    fig_tl.update_layout(
        height=120, margin=dict(l=20, r=20, t=10, b=20),
        xaxis=dict(range=[0, 52], title=None, showgrid=False, tickvals=list(range(0, 53, 4)),
                   ticktext=[f"W{w}" for w in range(0, 53, 4)], tickfont=dict(color='#8b949e', size=10)),
        yaxis=dict(visible=False, range=[0, 1]),
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        hoverlabel=dict(bgcolor='#161b22', font_size=12, font_family='Outfit'),
    )
    st.plotly_chart(fig_tl, use_container_width=True, key="timeline_chart")

    # ── Active Scenarios ──────────────────────────────────────────────────────
    st.markdown("#### 📢 Active Environmental / Shock Events")
    if snapshot["active_scenarios"]:
        for sc in snapshot["active_scenarios"]:
            st.markdown(f"""
            <div class="scenario-alert">
                <div class="scenario-name">
                    ⚠️ {sc['name']}
                    <span class="scenario-severity">{sc['severity']:.0%}</span>
                </div>
                <div class="scenario-detail">Type: {sc['type'].replace('_', ' ').title()}</div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="glass-card" style="text-align:center; padding:16px;">
            <span style="font-size:1.6rem;">🌤️</span><br>
            <span style="color:var(--accent-green); font-weight:600;">Clear weather!</span>
            <span style="color:var(--text-secondary);"> No active disruption events.</span>
        </div>
        """, unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 2: INTELLIGENCE & PREDICTIVE MODELING
# ═════════════════════════════════════════════════════════════════════════════

with tabs[1]:
    st.markdown("### 🧠 LogisChain Intelligence Hub")
    st.markdown(
        '<span style="color:#8b949e;">Real-time credit risk analytics, covenant monitoring, and AI-powered scoring.</span>',
        unsafe_allow_html=True
    )
    st.markdown("")

    # ── Score Breakdown ───────────────────────────────────────────────────────
    st.markdown("#### 🏆 Score Breakdown — 1000-Point Framework")

    # Calculate score components (mirroring engine logic)
    profit = snapshot['player']['revenue'] - snapshot['player']['costs'] - snapshot['player']['penalties']
    fin_score = min(max(profit / 1_000_000 * 100, 0), 400)

    total_ships = snapshot['player']['completed_shipments'] + snapshot['player']['active_shipments']
    if total_ships > 0:
        completed = engine.player.completed_shipments
        damage_rate = sum(1 for s in completed if s.status == "damaged") / max(total_ships, 1)
        delay_rate = sum(1 for s in completed if s.delay_days > 0) / max(total_ships, 1)
        risk_score_val = 300 * (1.0 - damage_rate * 0.5 - delay_rate * 0.3)
    else:
        risk_score_val = 150

    if engine.player.completed_shipments:
        avg_delay = np.mean([s.delay_days for s in engine.player.completed_shipments])
        ops_score_val = max(200 - avg_delay * 5, 0)
    else:
        ops_score_val = 100

    dec_score_val = min(len(engine.player.decisions_log) * 2, 100)

    scol1, scol2 = st.columns([2, 1])

    with scol1:
        fig_score = go.Figure()

        categories = ['Financial<br>(400)', 'Risk Mgmt<br>(300)', 'Operations<br>(200)', 'Decisions<br>(100)']
        scores = [fin_score, risk_score_val, ops_score_val, dec_score_val]
        maxes = [400, 300, 200, 100]
        colors = ['#58a6ff', '#bc8cff', '#3fb950', '#d29922']

        fig_score.add_trace(go.Bar(
            y=categories, x=maxes, orientation='h',
            marker=dict(color='rgba(255,255,255,0.04)', line=dict(width=0)),
            name='Maximum', showlegend=False, hoverinfo='skip'
        ))
        fig_score.add_trace(go.Bar(
            y=categories, x=scores, orientation='h',
            marker=dict(color=colors, line=dict(width=0)),
            name='Your Score',
            text=[f'{s:.0f}/{m}' for s, m in zip(scores, maxes)],
            textposition='inside', textfont=dict(color='white', size=13, family='Outfit'),
        ))

        fig_score.update_layout(
            barmode='overlay', height=200,
            margin=dict(l=10, r=20, t=10, b=10),
            xaxis=dict(range=[0, 420], showgrid=False, visible=False),
            yaxis=dict(tickfont=dict(color='#e6edf3', size=12, family='Outfit'), autorange='reversed'),
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            legend=dict(font=dict(color='#8b949e')),
            showlegend=False
        )
        st.plotly_chart(fig_score, use_container_width=True, key="score_breakdown")

    with scol2:
        total = fin_score + risk_score_val + ops_score_val + dec_score_val
        st.markdown(f"""
        <div class="glass-card" style="text-align:center;">
            <div class="metric-label">TOTAL SCORE</div>
            <div class="metric-value" style="font-size:2.8rem;">{total:.0f}</div>
            <div style="color:var(--text-secondary); font-size:0.82rem; margin-top:4px;">out of 1,000 pts</div>
        </div>
        """, unsafe_allow_html=True)

        # Player vs AI comparison
        st.markdown(f"""
        <div class="glass-card" style="padding:16px;">
            <div class="metric-label" style="margin-bottom:10px;">YOU vs AI OPPONENT</div>
            <div class="sidebar-stat">
                <span class="stat-label">💰 Cash</span>
                <span class="stat-value" style="font-size:0.75rem;">${snapshot['player']['cash']:,.0f}</span>
            </div>
            <div class="sidebar-stat">
                <span class="stat-label">🤖 AI Cash</span>
                <span class="stat-value" style="font-size:0.75rem;">${snapshot['ai_opponent']['cash']:,.0f}</span>
            </div>
            <div class="sidebar-stat">
                <span class="stat-label">🏆 AI Score</span>
                <span class="stat-value" style="font-size:0.75rem;">{snapshot['ai_opponent']['score']:.0f} pts</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Credit Risk Stress Test ───────────────────────────────────────────────
    st.markdown("#### 🛠️ Interactive Credit Risk & Covenant Stress Test")

    with st.container(border=True):
        col_pred = st.columns(3)
        with col_pred[0]:
            test_cur = st.slider("Current Ratio", 0.5, 3.5, 1.5, key="stress_cr")
            test_de = st.slider("Debt-to-Equity Ratio", 0.1, 5.0, 1.2, key="stress_de")
        with col_pred[1]:
            test_otif = st.slider("OTIF Rate", 0.5, 1.0, 0.95, key="stress_otif")
            test_lead = st.slider("Mean Lead Time (Days)", 3.0, 30.0, 12.0, key="stress_lead")
        with col_pred[2]:
            test_ccc = st.slider("Current CCC (Days)", 10.0, 150.0, 60.0, key="stress_ccc")
            test_dso = st.slider("Days Sales Outstanding", 10.0, 90.0, 35.0, key="stress_dso")

        # Synthetic PD score (weighted heuristic)
        pd_score = (
            (1.0 - min(test_cur / 3.5, 1.0)) * 0.20 +
            min(test_de / 5.0, 1.0) * 0.20 +
            (1.0 - test_otif) * 0.15 +
            min(test_lead / 30.0, 1.0) * 0.15 +
            min(test_ccc / 150.0, 1.0) * 0.20 +
            min(test_dso / 90.0, 1.0) * 0.10
        )

        g_col1, g_col2 = st.columns(2)

        with g_col1:
            # Risk Gauge
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=pd_score * 100,
                number=dict(suffix="%", font=dict(size=36, color='#e6edf3', family='Outfit')),
                title=dict(text="Probability of Default (SC-PD)", font=dict(size=14, color='#8b949e', family='Outfit')),
                gauge=dict(
                    axis=dict(range=[0, 100], tickwidth=1, tickcolor='#30363d',
                              tickfont=dict(color='#8b949e', size=10)),
                    bar=dict(color='#58a6ff', thickness=0.3),
                    bgcolor='rgba(0,0,0,0)',
                    borderwidth=0,
                    steps=[
                        dict(range=[0, 25], color='rgba(63, 185, 80, 0.15)'),
                        dict(range=[25, 50], color='rgba(210, 153, 34, 0.15)'),
                        dict(range=[50, 75], color='rgba(248, 81, 73, 0.12)'),
                        dict(range=[75, 100], color='rgba(248, 81, 73, 0.25)'),
                    ],
                    threshold=dict(line=dict(color='#f85149', width=3), thickness=0.8, value=pd_score * 100)
                )
            ))
            fig_gauge.update_layout(
                height=220, margin=dict(l=30, r=30, t=50, b=10),
                paper_bgcolor='rgba(0,0,0,0)', font=dict(family='Outfit')
            )
            st.plotly_chart(fig_gauge, use_container_width=True, key="pd_gauge")

        with g_col2:
            # CCC Waterfall
            dio = test_ccc * 0.45
            dso = test_dso
            dpo = dio + dso - test_ccc

            fig_waterfall = go.Figure(go.Waterfall(
                orientation="v",
                measure=["relative", "relative", "relative", "total"],
                x=["DIO<br>(Inventory)", "DSO<br>(Receivables)", "DPO<br>(Payables)", "CCC<br>(Net)"],
                y=[dio, dso, -dpo, 0],
                connector=dict(line=dict(color='rgba(255,255,255,0.1)')),
                increasing=dict(marker=dict(color='#f85149')),
                decreasing=dict(marker=dict(color='#3fb950')),
                totals=dict(marker=dict(color='#58a6ff')),
                textposition="outside",
                text=[f"{dio:.0f}d", f"{dso:.0f}d", f"-{dpo:.0f}d", f"{test_ccc:.0f}d"],
                textfont=dict(color='#e6edf3', size=11, family='JetBrains Mono'),
            ))
            fig_waterfall.update_layout(
                title=dict(text="Cash Conversion Cycle Decomposition", font=dict(size=14, color='#8b949e', family='Outfit')),
                height=220, margin=dict(l=10, r=10, t=40, b=10),
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                yaxis=dict(showgrid=False, tickfont=dict(color='#8b949e', size=10)),
                xaxis=dict(tickfont=dict(color='#e6edf3', size=11, family='Outfit')),
                showlegend=False
            )
            st.plotly_chart(fig_waterfall, use_container_width=True, key="ccc_waterfall")

        # Covenant status
        if test_ccc > 75.0:
            st.error(f"🚨 COVENANT BREACH: Cash Conversion Cycle ({test_ccc:.1f} days) exceeds 75-day risk limit!")
        elif test_ccc > 60.0:
            st.warning(f"⚠️ COVENANT WARNING: CCC ({test_ccc:.1f} days) approaching 75-day threshold.")
        else:
            st.success(f"✅ Covenant Active: CCC ({test_ccc:.1f} days) within safety thresholds.")

        if pd_score > 0.5:
            st.error(f"🚨 HIGH DEFAULT RISK: Estimated PD = {pd_score*100:.1f}% — immediate credit review recommended.")
        elif pd_score > 0.25:
            st.warning(f"⚠️ ELEVATED RISK: Estimated PD = {pd_score*100:.1f}% — enhanced monitoring recommended.")

    # ── Model Card ────────────────────────────────────────────────────────────
    if os.path.exists("data/models/model_card_scpd.md"):
        with st.expander("📄 View SC-PD Model Card"):
            with open("data/models/model_card_scpd.md", "r", encoding="utf-8") as f:
                st.markdown(f.read())

    # ── Historical Performance ────────────────────────────────────────────────
    if len(st.session_state.history) > 2:
        st.markdown("---")
        st.markdown("#### 📈 Performance Over Time")

        hist_df = pd.DataFrame([
            {
                "Week": h["week"],
                "Cash": h["player"]["cash"],
                "Score": h["player"]["score"],
                "AI Score": h["ai_opponent"]["score"],
            }
            for h in st.session_state.history
        ])

        hcol1, hcol2 = st.columns(2)

        with hcol1:
            fig_cash = go.Figure()
            fig_cash.add_trace(go.Scatter(
                x=hist_df["Week"], y=hist_df["Cash"],
                mode='lines+markers',
                line=dict(color='#58a6ff', width=2.5),
                marker=dict(size=5, color='#58a6ff'),
                fill='tozeroy', fillcolor='rgba(88, 166, 255, 0.06)',
                name='Cash Position'
            ))
            fig_cash.update_layout(
                title=dict(text="Cash Position History", font=dict(size=14, color='#8b949e', family='Outfit')),
                height=250, margin=dict(l=10, r=10, t=40, b=10),
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(showgrid=False, title=None, tickfont=dict(color='#8b949e', size=10)),
                yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.04)',
                           tickfont=dict(color='#8b949e', size=10), title=None),
                showlegend=False
            )
            st.plotly_chart(fig_cash, use_container_width=True, key="cash_history")

        with hcol2:
            fig_scores = go.Figure()
            fig_scores.add_trace(go.Scatter(
                x=hist_df["Week"], y=hist_df["Score"],
                mode='lines+markers',
                line=dict(color='#bc8cff', width=2.5),
                marker=dict(size=5, color='#bc8cff'),
                name='Your Score'
            ))
            fig_scores.add_trace(go.Scatter(
                x=hist_df["Week"], y=hist_df["AI Score"],
                mode='lines+markers',
                line=dict(color='#d29922', width=2, dash='dash'),
                marker=dict(size=4, color='#d29922'),
                name='AI Score'
            ))
            fig_scores.update_layout(
                title=dict(text="Score Comparison", font=dict(size=14, color='#8b949e', family='Outfit')),
                height=250, margin=dict(l=10, r=10, t=40, b=10),
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(showgrid=False, title=None, tickfont=dict(color='#8b949e', size=10)),
                yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.04)',
                           tickfont=dict(color='#8b949e', size=10), title=None),
                legend=dict(font=dict(color='#8b949e', size=11, family='Outfit'),
                            bgcolor='rgba(0,0,0,0)', x=0.02, y=0.98)
            )
            st.plotly_chart(fig_scores, use_container_width=True, key="score_history")


# ═════════════════════════════════════════════════════════════════════════════
# TAB 3: DYNAMIC SUPPLY CHAIN NETWORK
# ═════════════════════════════════════════════════════════════════════════════

with tabs[2]:
    st.markdown("### 🌐 Supply Chain Network Topology")
    st.markdown(
        '<span style="color:#8b949e;">Physical layer simulation mapping 100+ nodes and 500+ transportation corridors.</span>',
        unsafe_allow_html=True
    )
    st.markdown("")

    node_sum = engine.get_network_summary()

    # ── Network Stats Cards ───────────────────────────────────────────────────
    nc = node_sum['node_counts']
    net_cards = st.columns(5)
    node_types = [
        ("🏭", "Suppliers", nc.get("supplier", 0), "#58a6ff"),
        ("⚙️", "Manufacturers", nc.get("manufacturer", 0), "#bc8cff"),
        ("🚢", "Ports", nc.get("port", 0), "#3fb950"),
        ("📦", "Warehouses", nc.get("warehouse", 0), "#d29922"),
        ("🏪", "Customers", nc.get("customer", 0), "#f778ba"),
    ]

    for col, (icon, label, count, color) in zip(net_cards, node_types):
        with col:
            st.markdown(f"""
            <div class="metric-card" style="text-align:center;">
                <span style="font-size:1.8rem;">{icon}</span>
                <div class="metric-label" style="margin-top:6px;">{label}</div>
                <div class="metric-value" style="font-size:2rem;">{count}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("")

    # ── Charts Row ────────────────────────────────────────────────────────────
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        # Node Type Donut
        node_df = pd.DataFrame({
            "Type": list(nc.keys()),
            "Count": list(nc.values())
        })
        node_df["Type"] = node_df["Type"].str.title()

        fig_donut = px.pie(
            node_df, values="Count", names="Type",
            hole=0.55,
            color_discrete_sequence=['#58a6ff', '#bc8cff', '#3fb950', '#d29922', '#f778ba']
        )
        fig_donut.update_traces(
            textposition='inside', textinfo='percent+label',
            textfont=dict(size=12, family='Outfit', color='white'),
            marker=dict(line=dict(color='#0d1117', width=2))
        )
        fig_donut.update_layout(
            title=dict(text="Node Distribution by Type", font=dict(size=14, color='#8b949e', family='Outfit')),
            height=320, margin=dict(l=10, r=10, t=40, b=10),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            legend=dict(font=dict(color='#8b949e', size=11, family='Outfit'),
                        bgcolor='rgba(0,0,0,0)'),
            annotations=[dict(text=f"<b>{node_sum['total_nodes']}</b><br>Nodes",
                              x=0.5, y=0.5, font_size=16, font_color='#e6edf3',
                              font_family='Outfit', showarrow=False)]
        )
        st.plotly_chart(fig_donut, use_container_width=True, key="node_donut")

    with chart_col2:
        # Modality Bar Chart
        mod_data = node_sum['modality_counts']
        mod_df = pd.DataFrame({
            "Modality": [m.title() for m in mod_data.keys()],
            "Corridors": list(mod_data.values())
        })

        modality_colors = {'Ocean': '#58a6ff', 'Air': '#bc8cff', 'Rail': '#d29922', 'Road': '#3fb950'}

        fig_mod = go.Figure()
        for _, row in mod_df.iterrows():
            fig_mod.add_trace(go.Bar(
                x=[row["Modality"]], y=[row["Corridors"]],
                marker=dict(
                    color=modality_colors.get(row["Modality"], '#8b949e'),
                    line=dict(width=0),
                    cornerradius=6
                ),
                text=[str(row["Corridors"])],
                textposition='outside',
                textfont=dict(color='#e6edf3', size=13, family='JetBrains Mono'),
                name=row["Modality"], showlegend=False
            ))

        fig_mod.update_layout(
            title=dict(text="Transportation Corridors by Modality", font=dict(size=14, color='#8b949e', family='Outfit')),
            height=320, margin=dict(l=10, r=10, t=40, b=10),
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(showgrid=False, tickfont=dict(color='#e6edf3', size=12, family='Outfit')),
            yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.04)',
                       tickfont=dict(color='#8b949e', size=10), title=None),
        )
        st.plotly_chart(fig_mod, use_container_width=True, key="modality_bar")

    st.markdown("---")

    # ── Risk Heatmap — Top 15 Highest Risk Nodes ──────────────────────────────
    risk_col1, risk_col2 = st.columns([3, 2])

    with risk_col1:
        st.markdown("#### ⚠️ Top 15 Highest-Risk Nodes")
        risk_nodes = sorted(engine.nodes.values(), key=lambda n: n.risk_score, reverse=True)[:15]
        risk_rows = []
        for n in risk_nodes:
            risk_rows.append({
                "Node ID": n.node_id,
                "Name": n.name,
                "Type": n.node_type.title(),
                "Country": n.country,
                "Risk Score": f"{n.risk_score:.2f}",
                "Operational": "✅" if n.operational else "❌",
        })
        st.dataframe(pd.DataFrame(risk_rows), use_container_width=True, hide_index=True)

    with risk_col2:
        st.markdown("#### 📊 Network Health")

        op_nodes = sum(1 for n in engine.nodes.values() if n.operational)
        total_n = len(engine.nodes)
        disrupted_l = sum(1 for l in engine.links.values() if l.current_congestion > 1.5)
        total_l = len(engine.links)

        # Network health gauge
        health_pct = (op_nodes / max(total_n, 1)) * 100
        fig_health = go.Figure(go.Indicator(
            mode="gauge+number",
            value=health_pct,
            number=dict(suffix="%", font=dict(size=32, color='#e6edf3', family='Outfit')),
            gauge=dict(
                axis=dict(range=[0, 100], tickwidth=1, tickcolor='#30363d',
                          tickfont=dict(color='#8b949e', size=10)),
                bar=dict(color='#3fb950', thickness=0.3),
                bgcolor='rgba(0,0,0,0)', borderwidth=0,
                steps=[
                    dict(range=[0, 60], color='rgba(248, 81, 73, 0.12)'),
                    dict(range=[60, 85], color='rgba(210, 153, 34, 0.10)'),
                    dict(range=[85, 100], color='rgba(63, 185, 80, 0.10)'),
                ],
            )
        ))
        fig_health.update_layout(
            height=200, margin=dict(l=30, r=30, t=20, b=10),
            paper_bgcolor='rgba(0,0,0,0)',
        )
        st.plotly_chart(fig_health, use_container_width=True, key="health_gauge")

        st.markdown(f"""
        <div class="sidebar-stat">
            <span class="stat-label">Operational Nodes</span>
            <span class="stat-value">{op_nodes} / {total_n}</span>
        </div>
        <div class="sidebar-stat">
            <span class="stat-label">Disrupted Links</span>
            <span class="stat-value-accent">{disrupted_l} / {total_l}</span>
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# 6. Footer
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="app-footer">
    <span class="footer-brand">⛓️ LogisChain AI</span> — Predictive Trade Finance & Logistics Valuation Platform<br>
    <span style="font-size:0.7rem;">Built with Streamlit · Plotly · NetworkX · PyTorch</span>
</div>
""", unsafe_allow_html=True)
