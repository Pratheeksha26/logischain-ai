import random
import numpy as np
from typing import List
from .engine import ScenarioEvent

def get_default_scenarios() -> List[ScenarioEvent]:
    """
    Generate 5 mandatory shock scenarios with multi-domain financial impact.
    
    1. Suez Canal Blockage (ocean transit delay)
    2. Carrier Bankruptcy (severe logistic default risk)
    3. Major Port Strike (global supply chain congestion)
    4. Sudden Demand Shock (inventory and financial volatility)
    5. Geopolitical/Regulatory Change (tariffs and trade barriers)
    """
    scenarios = []

    # Scenario 1: Suez Canal Blockage
    scenarios.append(ScenarioEvent(
        event_id="SCEN_001",
        name="Suez Canal Blockage",
        description="A major container vessel has run aground in the Suez Canal, halting all maritime traffic and triggering massive detours and spiking fuel costs.",
        trigger_turn=10,
        duration_turns=6,
        event_type="canal_blockage",
        affected_links=[],  # Filled dynamically by modality 'ocean' during active phase
        severity=0.85,
        financial_impacts={
            "cost_increase_pct": 0.35,
            "delay_increase_days": 14
        }
    ))

    # Scenario 2: Ocean Carrier Bankruptcy
    scenarios.append(ScenarioEvent(
        event_id="SCEN_002",
        name="Global Carrier Bankruptcy",
        description="One of the largest ocean freight carriers has abruptly filed for Chapter 11 bankruptcy, leaving millions of dollars of cargo stranded at sea or impounded at ports.",
        trigger_turn=20,
        duration_turns=8,
        event_type="carrier_bankruptcy",
        affected_nodes=[],
        severity=0.90,
        financial_impacts={
            "loss_probability": 0.15,
            "cost_increase_pct": 0.50
        }
    ))

    # Scenario 3: West Coast Port Strike
    scenarios.append(ScenarioEvent(
        event_id="SCEN_003",
        name="Major Port Workers Strike",
        description="Dockworkers have initiated a labor strike at several critical shipping terminals, halting offloading operations and creating severe port backlogs.",
        trigger_turn=30,
        duration_turns=5,
        event_type="port_congestion",
        affected_nodes=["PRT_006", "PRT_007", "PRT_008"],  # Los Angeles, Long Beach, New York, etc.
        severity=0.80,
        financial_impacts={
            "delay_increase_days": 10,
            "demurrage_fee_usd": 15000
        }
    ))

    # Scenario 4: Post-Pandemic Demand Shock
    scenarios.append(ScenarioEvent(
        event_id="SCEN_004",
        name="Sudden Consumer Demand Shock",
        description="An unexpected economic downturn has drastically reduced consumer spending, causing orders to freeze and building massive inventory gluts.",
        trigger_turn=40,
        duration_turns=6,
        event_type="demand_shock",
        affected_nodes=[],
        severity=0.75,
        financial_impacts={
            "revenue_decrease_pct": 0.40,
            "inventory_carrying_cost_pct": 0.20
        }
    ))

    # Scenario 5: Geopolitical Regulatory Shift
    scenarios.append(ScenarioEvent(
        event_id="SCEN_005",
        name="Geopolitical Tariff Escalation",
        description="A major trade war has erupted, introducing sudden 25% tariffs on industrial components and demanding intensive custom audits and declarations.",
        trigger_turn=48,
        duration_turns=4,
        event_type="regulatory_change",
        affected_nodes=[],
        severity=0.70,
        financial_impacts={
            "tariff_fee_pct": 0.25,
            "audit_delay_days": 5
        }
    ))

    return scenarios
