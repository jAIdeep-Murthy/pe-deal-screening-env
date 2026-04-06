"""Pydantic models for the PE Deal Screening OpenEnv environment."""
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Shared / base models
# ---------------------------------------------------------------------------

class DealTeaser(BaseModel):
    """A single inbound deal teaser."""
    company_name: str
    sector: str
    revenue_mm: float = Field(..., description="Revenue in $MM")
    ebitda_mm: float = Field(..., description="EBITDA in $MM")
    ebitda_margin_pct: float = Field(..., description="EBITDA margin %")
    revenue_growth_pct: float = Field(..., description="YoY revenue growth %")
    asking_ev_ebitda: float = Field(..., description="Asking EV/EBITDA multiple")
    net_debt_mm: float = Field(..., description="Net debt in $MM")
    net_debt_ebitda: float = Field(..., description="Net Debt / EBITDA")
    founded_year: int
    hq_country: str
    deal_context: Optional[str] = None  # extra narrative for IC memo task


class PortfolioDeal(BaseModel):
    """A deal within a portfolio prioritization task."""
    company_name: str
    sector: str
    revenue_mm: float
    ebitda_mm: float
    ebitda_margin_pct: float
    revenue_growth_pct: float
    asking_ev_ebitda: float
    net_debt_ebitda: float
    expected_irr_pct: float = Field(..., description="Expected IRR %")
    risk_score: float = Field(..., ge=0.0, le=1.0, description="Risk score 0-1")


# ---------------------------------------------------------------------------
# Observations (what the agent sees)
# ---------------------------------------------------------------------------

class DealScreeningObservation(BaseModel):
    task: Literal["deal_screening"] = "deal_screening"
    episode_id: str
    step: int = 0
    deal: DealTeaser


class ICMemoObservation(BaseModel):
    task: Literal["ic_memo"] = "ic_memo"
    episode_id: str
    step: int = 0
    deal: DealTeaser


class PortfolioPrioritizationObservation(BaseModel):
    task: Literal["portfolio_prioritization"] = "portfolio_prioritization"
    episode_id: str
    step: int = 0
    portfolio: List[PortfolioDeal]
    fund_size_mm: float = 100.0
    max_single_pct: float = 40.0
    min_position_pct: float = 10.0


# ---------------------------------------------------------------------------
# Actions (what the agent returns)
# ---------------------------------------------------------------------------

class DealScreeningAction(BaseModel):
    decision: Literal["INVEST", "PASS"]
    confidence: float = Field(..., ge=0.0, le=1.0)
    rationale: str = Field(..., min_length=20)


class ICMemoAction(BaseModel):
    executive_summary: str = Field(..., min_length=50)
    investment_thesis: str = Field(..., min_length=50)
    key_risks: str = Field(..., min_length=30)
    financial_highlights: str = Field(..., min_length=30)
    recommendation: Literal["INVEST", "PASS", "CONDITIONAL"]


class AllocationItem(BaseModel):
    company_name: str
    allocation_mm: float = Field(..., ge=0.0)
    allocation_pct: float = Field(..., ge=0.0, le=100.0)
    rationale: str


class PortfolioPrioritizationAction(BaseModel):
    allocations: List[AllocationItem]
    total_deployed_mm: float
    rationale: str = Field(..., min_length=50)


# ---------------------------------------------------------------------------
# Step response
# ---------------------------------------------------------------------------

class StepResult(BaseModel):
    observation: Optional[Dict[str, Any]] = None
    reward: float
    done: bool
    info: Dict[str, Any] = {}


class ResetResponse(BaseModel):
    observation: Dict[str, Any]
    episode_id: str
    task: str
