from typing import Any, Dict, List, Optional
from openenv.core.env_server import Action, Observation, State


class TriageObservation(Observation):
    task_id: str = "deal_triage"
    company_name: str
    sector: str
    geography: str
    revenue_mm: float
    ebitda_mm: float
    ebitda_margin_pct: float
    leverage_x: float
    mandate_sectors: List[str]
    mandate_geographies: List[str]
    mandate_ebitda_min_mm: float
    mandate_ebitda_max_mm: float
    mandate_margin_min_pct: float
    mandate_leverage_max_x: float
    instructions: str = "Evaluate the deal against the fund mandate. Respond with JSON: {\"decision\": \"PASS|LIGHT_DD|DEEP_DIVE\", \"reason\": \"...\"}"


class TriageAction(Action):
    decision: str
    reason: str


class MemoObservation(Observation):
    task_id: str = "ic_memo"
    company_name: str
    sector: str
    geography: str
    revenue_y1_mm: float
    revenue_y2_mm: float
    revenue_y3_mm: float
    ebitda_y1_mm: float
    ebitda_y2_mm: float
    ebitda_y3_mm: float
    fcf_y3_mm: float
    market_notes: str
    risk_notes: str
    entry_ev_mm: float
    target_irr_pct: float
    required_positives: List[str]
    required_risks: List[str]
    instructions: str = "Write an IC memo with sections: ## Investment Thesis\n## Key Risks\n## Next Steps"


class MemoAction(Action):
    memo_text: str


class PortfolioObservation(Observation):
    task_id: str = "portfolio_prioritization"
    candidates: List[Dict[str, Any]]
    fund_size_mm: float
    max_single_deal_pct: float
    sector_cap_pct: float
    target_portfolio_irr_min: float
    target_portfolio_irr_max: float
    min_deals: int
    max_deals: int
    instructions: str = "Select deals satisfying constraints. Respond with JSON: {\"selected_deals\": [...], \"allocations\": {\"id\": 0.25}, \"rationale\": \"...\"}"


class PortfolioAction(Action):
    selected_deals: List[str]
    allocations: Dict[str, float]
    rationale: str


class DealState(State):
    task_id: str = ""
    seed: int = 0
    ground_truth: Optional[str] = None
    checklist_items: Optional[int] = None
    constraint_violations: int = 0
