import random
from typing import Dict, List, Optional
from .engine import SimulationEngine, PlayerState

def run_portfolio_management_mode(engine: SimulationEngine, risk_appetite: float = 0.5):
    """
    Trade Finance Portfolio Management Game Mode.
    
    The player manages a large portfolio of active LCs and shipments. The goal is to
    maximize portfolio value and cash position while maintaining low average defaults
    and high overall OTIF rates under uncertainty.
    
    Args:
        engine: The core simulation engine
        risk_appetite: How aggressive the AI/simulation should behave
    """
    # Adjust starting parameters
    engine.player.cash = 15_000_000.0  # Higher starting cash for portfolio management
    engine.ai_opponent.cash = 15_000_000.0
    
    # Generate an initial portfolio of letters of credit and active shipments
    suppliers = [n for n in engine.nodes.values() if n.node_type == "supplier"]
    ports = [n for n in engine.nodes.values() if n.node_type == "port"]
    
    for i in range(5):
        sup = suppliers[i % len(suppliers)]
        port = ports[i % len(ports)]
        val = 500_000.0 * (i + 1)
        
        # Player initial portfolio
        lc_p = engine.issue_lc(engine.player, sup.name, "Player Corp", val)
        engine.create_shipment(engine.player, sup.node_id, port.node_id, "Electronics", val, insure=True, lc=lc_p)
        
        # AI initial portfolio
        lc_a = engine.issue_lc(engine.ai_opponent, sup.name, "AI Corp", val)
        engine.create_shipment(engine.ai_opponent, sup.node_id, port.node_id, "Electronics", val, insure=False, lc=lc_a)

def run_scf_pricing_mode(engine: SimulationEngine, discount_rate_base: float = 0.05):
    """
    Supply Chain Finance (SCF) dynamic pricing mode.
    
    In this mode, players act as factors/funders offering dynamic discount rates
    to suppliers on their receivables. High supplier risk or low credit scores require
    higher discount margins to balance default likelihood.
    
    Args:
        engine: The core simulation engine
        discount_rate_base: Base discount rate charged by the funder
    """
    engine.player.cash = 5_000_000.0   # Tighter starting cash requiring strategic pricing
    engine.ai_opponent.cash = 5_000_000.0
    
    # Dynamic discount adjustments logic is applied inside streamlit app view during turns.
