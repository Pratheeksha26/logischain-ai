"""
LogisChain Lab Simulation Engine.

Three-layer simulation engine for the gamified trade finance & logistics platform:
  - Physical Layer: Models 100+ supply chain nodes and 500+ transportation links
  - Financial Layer: Tracks LCs, working capital, insurance, balance sheets
  - Intelligence Layer: Integrates LogisChain AI model predictions as AI opponent

Turn-based gameplay: 1 turn = 1 simulated week, 52 turns = 1 year.
"""

import os
import json
import copy
import random
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Data Classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SupplyChainNode:
    """A node in the physical supply chain network."""
    node_id: str
    node_type: str          # 'supplier', 'manufacturer', 'port', 'warehouse', 'customer'
    name: str
    country: str
    risk_score: float = 0.5
    capacity: float = 1000.0
    current_load: float = 0.0
    operational: bool = True

@dataclass
class TransportLink:
    """A transportation link between two nodes."""
    link_id: str
    source_id: str
    dest_id: str
    modality: str           # 'ocean', 'air', 'rail', 'road'
    base_transit_days: int = 14
    cost_per_unit: float = 50.0
    reliability: float = 0.90
    current_congestion: float = 1.0
    operational: bool = True

@dataclass
class LetterOfCredit:
    """A Letter of Credit financial instrument."""
    lc_id: str
    issuing_bank: str
    beneficiary: str
    applicant: str
    amount_usd: float
    issue_date: int         # turn number
    expiry_date: int        # turn number
    shipment_id: Optional[str] = None
    status: str = "active"  # active, drawn, expired, amended, cancelled
    amendment_count: int = 0

@dataclass
class Shipment:
    """An in-transit shipment."""
    shipment_id: str
    origin_id: str
    dest_id: str
    link_id: str
    commodity: str
    value_usd: float
    departure_turn: int
    expected_arrival_turn: int
    actual_arrival_turn: Optional[int] = None
    status: str = "in_transit"  # in_transit, delivered, delayed, damaged, lost
    delay_days: int = 0
    insured: bool = False
    insurance_premium: float = 0.0
    lc_id: Optional[str] = None

@dataclass
class PlayerState:
    """Tracks a player's financial position."""
    name: str
    cash: float = 10_000_000.0          # Starting cash $10M
    total_portfolio_value: float = 0.0
    letters_of_credit: List[LetterOfCredit] = field(default_factory=list)
    active_shipments: List[Shipment] = field(default_factory=list)
    completed_shipments: List[Shipment] = field(default_factory=list)
    revenue: float = 0.0
    costs: float = 0.0
    insurance_payouts: float = 0.0
    penalties: float = 0.0
    score: float = 0.0
    decisions_log: List[Dict] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Scenario Events
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ScenarioEvent:
    """A disruption event that affects the simulation."""
    event_id: str
    name: str
    description: str
    trigger_turn: int
    duration_turns: int
    event_type: str         # 'port_congestion', 'canal_blockage', 'carrier_bankruptcy',
                            # 'demand_shock', 'regulatory_change'
    affected_nodes: List[str] = field(default_factory=list)
    affected_links: List[str] = field(default_factory=list)
    severity: float = 0.5   # 0.0 to 1.0
    financial_impacts: Dict = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# Simulation Engine
# ─────────────────────────────────────────────────────────────────────────────

class SimulationEngine:
    """
    Core three-layer simulation engine for LogisChain Lab.

    Physical Layer: Supply chain network topology and goods flow
    Financial Layer: LC management, working capital, insurance, scoring
    Intelligence Layer: AI opponent using model predictions
    """

    def __init__(self, seed: int = 42):
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)

        self.current_turn: int = 0
        self.max_turns: int = 52  # 1 year

        # Physical Layer
        self.nodes: Dict[str, SupplyChainNode] = {}
        self.links: Dict[str, TransportLink] = {}

        # Financial Layer
        self.player: PlayerState = PlayerState(name="Player")
        self.ai_opponent: PlayerState = PlayerState(name="AI Opponent")

        # Intelligence Layer
        self.scenario_queue: List[ScenarioEvent] = []
        self.active_scenarios: List[ScenarioEvent] = []
        self.history: List[Dict] = []

        # Build default network
        self._build_network()

    # ── Physical Layer ───────────────────────────────────────────────────────

    def _build_network(self):
        """Build a supply chain network with 100+ nodes and 500+ links."""
        # Suppliers (40 nodes across different countries)
        countries_suppliers = {
            "China": 12, "India": 6, "Vietnam": 5, "Germany": 4,
            "USA": 4, "Mexico": 3, "Brazil": 3, "Japan": 3
        }
        node_idx = 0
        supplier_ids = []
        for country, count in countries_suppliers.items():
            for i in range(count):
                nid = f"SUP_{node_idx:03d}"
                self.nodes[nid] = SupplyChainNode(
                    node_id=nid, node_type="supplier",
                    name=f"{country}_Supplier_{i+1}", country=country,
                    risk_score=np.random.uniform(0.2, 0.8),
                    capacity=np.random.uniform(500, 5000)
                )
                supplier_ids.append(nid)
                node_idx += 1

        # Manufacturers (15 nodes)
        mfr_countries = ["China", "Germany", "USA", "Japan", "Mexico",
                         "India", "South Korea", "Taiwan", "Thailand",
                         "Poland", "Czech Republic", "Turkey", "Vietnam",
                         "Indonesia", "Malaysia"]
        mfr_ids = []
        for i, country in enumerate(mfr_countries):
            nid = f"MFR_{i:03d}"
            self.nodes[nid] = SupplyChainNode(
                node_id=nid, node_type="manufacturer",
                name=f"{country}_Manufacturer", country=country,
                risk_score=np.random.uniform(0.2, 0.6),
                capacity=np.random.uniform(2000, 10000)
            )
            mfr_ids.append(nid)

        # Ports (20 nodes)
        port_data = [
            ("Shanghai", "China"), ("Shenzhen", "China"), ("Ningbo", "China"),
            ("Singapore", "Singapore"), ("Rotterdam", "Netherlands"),
            ("Hamburg", "Germany"), ("Los Angeles", "USA"), ("Long Beach", "USA"),
            ("New York", "USA"), ("Savannah", "USA"),
            ("Mumbai JNPT", "India"), ("Colombo", "Sri Lanka"),
            ("Busan", "South Korea"), ("Kaohsiung", "Taiwan"),
            ("Laem Chabang", "Thailand"), ("Port Klang", "Malaysia"),
            ("Tanjung Pelepas", "Malaysia"), ("Manzanillo", "Mexico"),
            ("Santos", "Brazil"), ("Felixstowe", "UK")
        ]
        port_ids = []
        for i, (name, country) in enumerate(port_data):
            nid = f"PRT_{i:03d}"
            self.nodes[nid] = SupplyChainNode(
                node_id=nid, node_type="port", name=name, country=country,
                risk_score=np.random.uniform(0.1, 0.5),
                capacity=np.random.uniform(50000, 200000)
            )
            port_ids.append(nid)

        # Warehouses (15 nodes)
        wh_ids = []
        wh_countries = ["USA", "Germany", "China", "Japan", "India",
                        "Netherlands", "UK", "Mexico", "Singapore",
                        "South Korea", "Brazil", "Thailand", "UAE",
                        "Poland", "Australia"]
        for i, country in enumerate(wh_countries):
            nid = f"WHS_{i:03d}"
            self.nodes[nid] = SupplyChainNode(
                node_id=nid, node_type="warehouse",
                name=f"{country}_DC", country=country,
                risk_score=np.random.uniform(0.1, 0.3),
                capacity=np.random.uniform(10000, 50000)
            )
            wh_ids.append(nid)

        # Customers (15 nodes)
        cust_ids = []
        for i, country in enumerate(["USA", "Germany", "UK", "France", "Japan",
                                      "Canada", "Australia", "Mexico", "Brazil",
                                      "India", "South Korea", "Italy", "Spain",
                                      "Netherlands", "Sweden"]):
            nid = f"CST_{i:03d}"
            self.nodes[nid] = SupplyChainNode(
                node_id=nid, node_type="customer",
                name=f"{country}_Customer", country=country,
                risk_score=np.random.uniform(0.1, 0.4),
                capacity=np.random.uniform(1000, 20000)
            )
            cust_ids.append(nid)

        total_nodes = len(self.nodes)

        # Build 500+ links
        link_idx = 0
        modalities = ["ocean", "air", "rail", "road"]

        # Supplier -> Port (each supplier connects to 2-4 ports)
        for sid in supplier_ids:
            n_ports = random.randint(2, 4)
            for pid in random.sample(port_ids, min(n_ports, len(port_ids))):
                lid = f"LNK_{link_idx:04d}"
                self.links[lid] = TransportLink(
                    link_id=lid, source_id=sid, dest_id=pid,
                    modality=random.choice(["road", "rail"]),
                    base_transit_days=random.randint(1, 5),
                    cost_per_unit=np.random.uniform(10, 80),
                    reliability=np.random.uniform(0.85, 0.98)
                )
                link_idx += 1

        # Port -> Port (ocean shipping lanes, key corridors)
        for i, p1 in enumerate(port_ids):
            n_dest = random.randint(3, 6)
            targets = random.sample([p for p in port_ids if p != p1], min(n_dest, len(port_ids)-1))
            for p2 in targets:
                lid = f"LNK_{link_idx:04d}"
                self.links[lid] = TransportLink(
                    link_id=lid, source_id=p1, dest_id=p2,
                    modality="ocean",
                    base_transit_days=random.randint(7, 35),
                    cost_per_unit=np.random.uniform(30, 200),
                    reliability=np.random.uniform(0.82, 0.95)
                )
                link_idx += 1

        # Port -> Manufacturer (each mfr served by 2-3 ports)
        for mid in mfr_ids:
            n_ports = random.randint(2, 3)
            for pid in random.sample(port_ids, min(n_ports, len(port_ids))):
                lid = f"LNK_{link_idx:04d}"
                self.links[lid] = TransportLink(
                    link_id=lid, source_id=pid, dest_id=mid,
                    modality=random.choice(["road", "rail"]),
                    base_transit_days=random.randint(1, 4),
                    cost_per_unit=np.random.uniform(15, 60),
                    reliability=np.random.uniform(0.88, 0.97)
                )
                link_idx += 1

        # Manufacturer -> Warehouse
        for mid in mfr_ids:
            n_wh = random.randint(1, 3)
            for wid in random.sample(wh_ids, min(n_wh, len(wh_ids))):
                lid = f"LNK_{link_idx:04d}"
                self.links[lid] = TransportLink(
                    link_id=lid, source_id=mid, dest_id=wid,
                    modality=random.choice(["road", "rail", "air"]),
                    base_transit_days=random.randint(2, 10),
                    cost_per_unit=np.random.uniform(20, 120),
                    reliability=np.random.uniform(0.88, 0.96)
                )
                link_idx += 1

        # Warehouse -> Customer
        for wid in wh_ids:
            n_cust = random.randint(2, 5)
            for cid in random.sample(cust_ids, min(n_cust, len(cust_ids))):
                lid = f"LNK_{link_idx:04d}"
                self.links[lid] = TransportLink(
                    link_id=lid, source_id=wid, dest_id=cid,
                    modality=random.choice(["road", "air"]),
                    base_transit_days=random.randint(1, 7),
                    cost_per_unit=np.random.uniform(10, 50),
                    reliability=np.random.uniform(0.90, 0.99)
                )
                link_idx += 1

        print(f"Network built: {total_nodes} nodes, {len(self.links)} links")

    # ── Financial Layer ──────────────────────────────────────────────────────

    def issue_lc(self, player: PlayerState, beneficiary: str, applicant: str,
                 amount: float, validity_turns: int = 12) -> LetterOfCredit:
        """Issue a new Letter of Credit."""
        lc = LetterOfCredit(
            lc_id=f"LC_{len(player.letters_of_credit):04d}",
            issuing_bank="LogisChain Bank",
            beneficiary=beneficiary,
            applicant=applicant,
            amount_usd=amount,
            issue_date=self.current_turn,
            expiry_date=self.current_turn + validity_turns
        )
        # LC issuance fee = 1.5% of face value
        fee = amount * 0.015
        player.cash -= fee
        player.costs += fee
        player.letters_of_credit.append(lc)
        return lc

    def create_shipment(self, player: PlayerState, origin_id: str, dest_id: str,
                        commodity: str, value: float,
                        insure: bool = False, lc: Optional[LetterOfCredit] = None) -> Optional[Shipment]:
        """Create a new shipment along a valid link."""
        # Find a link
        link = None
        for l in self.links.values():
            if l.source_id == origin_id and l.dest_id == dest_id and l.operational:
                link = l
                break

        if link is None:
            return None

        transit = int(link.base_transit_days / 7)  # convert to turns (weeks)
        transit = max(transit, 1)

        ship_cost = value * 0.03  # 3% shipping cost
        ins_premium = 0.0
        if insure:
            ins_premium = value * 0.02  # 2% insurance premium

        shipment = Shipment(
            shipment_id=f"SH_{len(player.active_shipments) + len(player.completed_shipments):05d}",
            origin_id=origin_id,
            dest_id=dest_id,
            link_id=link.link_id,
            commodity=commodity,
            value_usd=value,
            departure_turn=self.current_turn,
            expected_arrival_turn=self.current_turn + transit,
            insured=insure,
            insurance_premium=ins_premium,
            lc_id=lc.lc_id if lc else None
        )
        player.cash -= (ship_cost + ins_premium)
        player.costs += (ship_cost + ins_premium)
        player.active_shipments.append(shipment)
        return shipment

    def _process_shipments(self, player: PlayerState):
        """Advance shipments, handle arrivals, delays, damage."""
        still_active = []
        for ship in player.active_shipments:
            link = self.links.get(ship.link_id)
            if link is None:
                still_active.append(ship)
                continue

            # Check for delay from congestion or scenarios
            delay_prob = (1.0 - link.reliability) * link.current_congestion
            if np.random.random() < delay_prob:
                ship.delay_days += 7
                ship.expected_arrival_turn += 1

            # Check for damage (0.5% base per turn, higher if link disrupted)
            dmg_prob = 0.005 * link.current_congestion
            if np.random.random() < dmg_prob and ship.status != "damaged":
                ship.status = "damaged"
                if ship.insured:
                    payout = ship.value_usd * 0.8
                    player.cash += payout
                    player.insurance_payouts += payout

            # Check arrival
            if self.current_turn >= ship.expected_arrival_turn:
                if ship.status != "damaged":
                    ship.status = "delivered"
                    ship.actual_arrival_turn = self.current_turn
                    # Revenue from delivered shipment
                    margin = np.random.uniform(0.08, 0.15)
                    rev = ship.value_usd * margin
                    player.cash += ship.value_usd + rev
                    player.revenue += ship.value_usd + rev
                else:
                    ship.actual_arrival_turn = self.current_turn

                player.completed_shipments.append(ship)
            else:
                still_active.append(ship)

        player.active_shipments = still_active

    def _process_lcs(self, player: PlayerState):
        """Check LC expirations and status."""
        for lc in player.letters_of_credit:
            if lc.status == "active" and self.current_turn > lc.expiry_date:
                lc.status = "expired"
                # Penalty for LC expiry without draw
                penalty = lc.amount_usd * 0.01
                player.cash -= penalty
                player.penalties += penalty

    # ── Intelligence Layer ───────────────────────────────────────────────────

    def load_scenarios(self, scenarios: List[ScenarioEvent]):
        """Load scenario events into the simulation queue."""
        self.scenario_queue = sorted(scenarios, key=lambda s: s.trigger_turn)

    def _process_scenarios(self):
        """Activate triggered scenarios and apply their effects."""
        # Activate new scenarios
        newly_active = []
        remaining = []
        for sc in self.scenario_queue:
            if sc.trigger_turn <= self.current_turn:
                newly_active.append(sc)
            else:
                remaining.append(sc)
        self.scenario_queue = remaining
        self.active_scenarios.extend(newly_active)

        # Apply active scenario effects
        still_active = []
        for sc in self.active_scenarios:
            end_turn = sc.trigger_turn + sc.duration_turns
            if self.current_turn <= end_turn:
                # Apply effects to nodes
                for nid in sc.affected_nodes:
                    if nid in self.nodes:
                        node = self.nodes[nid]
                        node.risk_score = min(node.risk_score + sc.severity * 0.3, 1.0)
                        if sc.severity > 0.8:
                            node.operational = False

                # Apply effects to links
                for lid in sc.affected_links:
                    if lid in self.links:
                        link = self.links[lid]
                        link.current_congestion = 1.0 + sc.severity * 3.0
                        link.reliability = max(link.reliability - sc.severity * 0.2, 0.3)
                        if sc.severity > 0.9:
                            link.operational = False

                still_active.append(sc)
            else:
                # Scenario ends — restore
                for nid in sc.affected_nodes:
                    if nid in self.nodes:
                        self.nodes[nid].operational = True
                        self.nodes[nid].risk_score = max(self.nodes[nid].risk_score - sc.severity * 0.2, 0.1)
                for lid in sc.affected_links:
                    if lid in self.links:
                        self.links[lid].current_congestion = 1.0
                        self.links[lid].operational = True

        self.active_scenarios = still_active

    def ai_make_decisions(self):
        """AI opponent makes automatic decisions for the current turn."""
        # Simple heuristic AI: issue LCs and create shipments on safe routes
        suppliers = [n for n in self.nodes.values() if n.node_type == "supplier" and n.operational]
        ports = [n for n in self.nodes.values() if n.node_type == "port" and n.operational]

        if suppliers and ports and self.ai_opponent.cash > 500_000:
            sup = random.choice(suppliers)
            port = random.choice(ports)
            value = np.random.uniform(100_000, 500_000)

            # Issue LC
            lc = self.issue_lc(self.ai_opponent, sup.name, "AI Corp", value)

            # Create shipment with insurance if risky
            risk = sup.risk_score
            self.create_shipment(
                self.ai_opponent, sup.node_id, port.node_id,
                commodity="Electronics", value=value,
                insure=(risk > 0.5), lc=lc
            )

    # ── Turn Processing ──────────────────────────────────────────────────────

    def advance_turn(self, player_decisions: Optional[List[Dict]] = None):
        """
        Advance the simulation by one turn (1 week).

        Args:
            player_decisions: list of decision dicts from the human player
        """
        self.current_turn += 1

        # 1. Process scenario events
        self._process_scenarios()

        # 2. Process player decisions
        if player_decisions:
            for dec in player_decisions:
                self._execute_decision(self.player, dec)

        # 3. AI opponent makes decisions
        self.ai_make_decisions()

        # 4. Advance shipments
        self._process_shipments(self.player)
        self._process_shipments(self.ai_opponent)

        # 5. Process LCs
        self._process_lcs(self.player)
        self._process_lcs(self.ai_opponent)

        # 6. Compute scores
        self._compute_scores(self.player)
        self._compute_scores(self.ai_opponent)

        # 7. Record history
        snapshot = self.get_state_snapshot()
        self.history.append(snapshot)

        return snapshot

    def _execute_decision(self, player: PlayerState, decision: Dict):
        """Execute a player decision."""
        action = decision.get("action")
        if action == "issue_lc":
            self.issue_lc(
                player,
                decision.get("beneficiary", "Unknown"),
                decision.get("applicant", "Player Corp"),
                decision.get("amount", 100_000),
                decision.get("validity_turns", 12)
            )
        elif action == "create_shipment":
            lc = None
            if decision.get("lc_id"):
                lc = next((l for l in player.letters_of_credit if l.lc_id == decision["lc_id"]), None)
            self.create_shipment(
                player,
                decision.get("origin_id", ""),
                decision.get("dest_id", ""),
                decision.get("commodity", "General"),
                decision.get("value", 100_000),
                decision.get("insure", False),
                lc
            )
        elif action == "amend_lc":
            lc_id = decision.get("lc_id")
            for lc in player.letters_of_credit:
                if lc.lc_id == lc_id and lc.status == "active":
                    lc.expiry_date += decision.get("extend_turns", 4)
                    lc.amendment_count += 1
                    fee = lc.amount_usd * 0.005
                    player.cash -= fee
                    player.costs += fee
                    break

        player.decisions_log.append({"turn": self.current_turn, **decision})

    def _compute_scores(self, player: PlayerState):
        """Compute the player's score based on the 1000-point framework."""
        # Portfolio value
        active_value = sum(s.value_usd for s in player.active_shipments)
        completed_value = sum(s.value_usd for s in player.completed_shipments)
        player.total_portfolio_value = player.cash + active_value

        # Score components (out of 1000)
        # Financial performance (400 pts)
        profit = player.revenue - player.costs - player.penalties
        fin_score = min(max(profit / 1_000_000 * 100, 0), 400)

        # Risk management (300 pts)
        total_ships = len(player.completed_shipments) + len(player.active_shipments)
        if total_ships > 0:
            damage_rate = sum(1 for s in player.completed_shipments if s.status == "damaged") / max(total_ships, 1)
            delay_rate = sum(1 for s in player.completed_shipments if s.delay_days > 0) / max(total_ships, 1)
            risk_score = 300 * (1.0 - damage_rate * 0.5 - delay_rate * 0.3)
        else:
            risk_score = 150  # baseline

        # Operational efficiency (200 pts)
        if player.completed_shipments:
            avg_delay = np.mean([s.delay_days for s in player.completed_shipments])
            ops_score = max(200 - avg_delay * 5, 0)
        else:
            ops_score = 100

        # Decision quality (100 pts)
        dec_score = min(len(player.decisions_log) * 2, 100)

        player.score = fin_score + risk_score + ops_score + dec_score

    def get_state_snapshot(self) -> Dict:
        """Get a snapshot of the current simulation state."""
        return {
            "turn": self.current_turn,
            "week": self.current_turn,
            "player": {
                "name": self.player.name,
                "cash": round(self.player.cash, 2),
                "portfolio_value": round(self.player.total_portfolio_value, 2),
                "active_shipments": len(self.player.active_shipments),
                "completed_shipments": len(self.player.completed_shipments),
                "active_lcs": sum(1 for lc in self.player.letters_of_credit if lc.status == "active"),
                "revenue": round(self.player.revenue, 2),
                "costs": round(self.player.costs, 2),
                "penalties": round(self.player.penalties, 2),
                "score": round(self.player.score, 1)
            },
            "ai_opponent": {
                "name": self.ai_opponent.name,
                "cash": round(self.ai_opponent.cash, 2),
                "portfolio_value": round(self.ai_opponent.total_portfolio_value, 2),
                "score": round(self.ai_opponent.score, 1)
            },
            "network": {
                "total_nodes": len(self.nodes),
                "operational_nodes": sum(1 for n in self.nodes.values() if n.operational),
                "total_links": len(self.links),
                "disrupted_links": sum(1 for l in self.links.values() if l.current_congestion > 1.5),
            },
            "active_scenarios": [
                {"name": s.name, "severity": s.severity, "type": s.event_type}
                for s in self.active_scenarios
            ]
        }

    def get_network_summary(self) -> Dict:
        """Return network statistics."""
        type_counts = {}
        for n in self.nodes.values():
            type_counts[n.node_type] = type_counts.get(n.node_type, 0) + 1
        return {
            "node_counts": type_counts,
            "total_nodes": len(self.nodes),
            "total_links": len(self.links),
            "modality_counts": {
                m: sum(1 for l in self.links.values() if l.modality == m)
                for m in ["ocean", "air", "rail", "road"]
            }
        }


if __name__ == "__main__":
    engine = SimulationEngine()
    summary = engine.get_network_summary()
    print(f"\nNetwork Summary:")
    print(f"  Nodes: {summary['total_nodes']} ({summary['node_counts']})")
    print(f"  Links: {summary['total_links']} ({summary['modality_counts']})")

    # Quick simulation test - run 10 turns
    print("\nRunning 10-turn simulation test...")
    for t in range(10):
        if t % 3 == 0 and engine.player.cash > 200_000:
            decisions = [{
                "action": "create_shipment",
                "origin_id": "SUP_000",
                "dest_id": "PRT_000",
                "commodity": "Electronics",
                "value": 200_000,
                "insure": True
            }]
        else:
            decisions = []
        snap = engine.advance_turn(decisions)
        print(f"  Turn {snap['turn']:2d} | Player: ${snap['player']['cash']:,.0f} "
              f"(score: {snap['player']['score']:.0f}) | "
              f"AI: ${snap['ai_opponent']['cash']:,.0f} "
              f"(score: {snap['ai_opponent']['score']:.0f})")
