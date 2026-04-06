"""Deterministic synthetic deal generator for PE Deal Screening environment."""
import random
from typing import Any, Dict, List, Optional, Tuple

from pe_env.models import DealTeaser, PortfolioDeal

SECTORS = [
    "Healthcare", "Technology", "Business Services", "Industrials",
    "Consumer", "Financial Services", "Energy", "Education",
]

COUNTRIES = ["USA", "UK", "Germany", "France", "Canada", "Australia", "Netherlands"]

COMPANY_NAMES = [
    "Apex Analytics", "Bridgepoint Solutions", "Crestview Systems",
    "Delta MedTech", "Evergreen Software", "Falcon Industrials",
    "Granite Business Services", "Harbour Capital", "Ironclad Tech",
    "Junction Healthcare", "Keystone Education", "Landmark Consumer",
    "Meridian Financial", "Northstar Energy", "Oaktree Logistics",
    "Pinnacle Business Services", "Quantum Technology", "Ridgeline Healthcare",
    "Summit Software", "Thornberry Industrials", "Unified Analytics",
    "Vantage Consumer", "Westbrook Education", "Xcel Business Services",
    "Yellowstone Energy", "Zenith Technology",
]

DEAL_CONTEXTS = [
    "Founder-led business seeking growth capital; management team has 15+ years experience.",
    "PE-backed company pursuing a buy-and-build strategy in a fragmented market.",
    "Corporate carve-out with strong standalone prospects and clean financials.",
    "Family-owned business in transition; second-generation management.",
    "Tech-enabled services platform with recurring revenue model.",
    "Asset-light distribution business with strong FCF conversion.",
    "SaaS platform serving mid-market enterprise clients with 95%+ gross retention.",
    "Healthcare services platform benefiting from favorable demographic tailwinds.",
]


def _rng(seed: int) -> random.Random:
    return random.Random(seed)


def generate_deal(seed: int) -> DealTeaser:
    """Generate a single deterministic deal teaser from a seed."""
    rng = _rng(seed)
    sector = rng.choice(SECTORS)
    revenue_mm = round(rng.uniform(20, 500), 1)
    ebitda_margin_pct = round(rng.uniform(8, 40), 1)
    ebitda_mm = round(revenue_mm * ebitda_margin_pct / 100, 1)
    revenue_growth_pct = round(rng.uniform(-5, 35), 1)
    asking_ev_ebitda = round(rng.uniform(5, 20), 1)
    net_debt_ebitda = round(rng.uniform(0, 6), 2)
    net_debt_mm = round(ebitda_mm * net_debt_ebitda, 1)
    founded_year = rng.randint(1985, 2018)
    company_name = rng.choice(COMPANY_NAMES) + f" {rng.randint(1, 99)}"
    hq_country = rng.choice(COUNTRIES)
    deal_context = rng.choice(DEAL_CONTEXTS)

    return DealTeaser(
        company_name=company_name,
        sector=sector,
        revenue_mm=revenue_mm,
        ebitda_mm=ebitda_mm,
        ebitda_margin_pct=ebitda_margin_pct,
        revenue_growth_pct=revenue_growth_pct,
        asking_ev_ebitda=asking_ev_ebitda,
        net_debt_mm=net_debt_mm,
        net_debt_ebitda=net_debt_ebitda,
        founded_year=founded_year,
        hq_country=hq_country,
        deal_context=deal_context,
    )


def compute_ground_truth_decision(deal: DealTeaser) -> str:
    """Rule-based ground truth for deal screening (INVEST or PASS)."""
    # Hard pass criteria
    if deal.ebitda_margin_pct < 10:
        return "PASS"
    if deal.net_debt_ebitda > 5.0:
        return "PASS"
    if deal.asking_ev_ebitda > 16:
        return "PASS"
    if deal.revenue_growth_pct < 0:
        return "PASS"
    # Green-light criteria
    if (
        deal.ebitda_margin_pct >= 15
        and deal.revenue_growth_pct >= 8
        and deal.asking_ev_ebitda <= 12
        and deal.net_debt_ebitda <= 3.5
    ):
        return "INVEST"
    # Borderline - use scoring
    score = 0
    if deal.ebitda_margin_pct >= 12:
        score += 1
    if deal.revenue_growth_pct >= 5:
        score += 1
    if deal.asking_ev_ebitda <= 13:
        score += 1
    if deal.net_debt_ebitda <= 4.0:
        score += 1
    return "INVEST" if score >= 3 else "PASS"


def generate_portfolio(seed: int, n: int = 5) -> List[PortfolioDeal]:
    """Generate a portfolio of n deals for prioritization task."""
    rng = _rng(seed)
    portfolio = []
    sectors_used = set()
    for i in range(n):
        s = seed * 100 + i
        deal = generate_deal(s)
        expected_irr = round(rng.uniform(12, 35), 1)
        risk_score = round(rng.uniform(0.1, 0.9), 2)
        # Ensure at least some sector diversity
        sector = deal.sector
        if sector in sectors_used and len(sectors_used) < len(SECTORS):
            remaining = [s for s in SECTORS if s not in sectors_used]
            if remaining:
                sector = rng.choice(remaining)
        sectors_used.add(sector)
        portfolio.append(
            PortfolioDeal(
                company_name=deal.company_name,
                sector=sector,
                revenue_mm=deal.revenue_mm,
                ebitda_mm=deal.ebitda_mm,
                ebitda_margin_pct=deal.ebitda_margin_pct,
                revenue_growth_pct=deal.revenue_growth_pct,
                asking_ev_ebitda=deal.asking_ev_ebitda,
                net_debt_ebitda=deal.net_debt_ebitda,
                expected_irr_pct=expected_irr,
                risk_score=risk_score,
            )
        )
    return portfolio


def compute_optimal_allocation(portfolio: List[PortfolioDeal], fund_size_mm: float = 100.0) -> Dict[str, float]:
    """Compute a heuristic optimal allocation (risk-adjusted IRR weighting)."""
    scores = []
    for deal in portfolio:
        # Risk-adjusted IRR: higher IRR and lower risk = higher score
        risk_adj = deal.expected_irr_pct * (1 - deal.risk_score)
        scores.append(max(risk_adj, 0))
    total = sum(scores)
    if total == 0:
        weights = [1 / len(portfolio)] * len(portfolio)
    else:
        weights = [s / total for s in scores]
    # Apply constraints: max 40%, min 10% if included
    alloc_pct = [w * 100 for w in weights]
    # Cap at 40%
    alloc_pct = [min(p, 40.0) for p in alloc_pct]
    # Renormalize
    total_pct = sum(alloc_pct)
    alloc_pct = [p * 100 / total_pct for p in alloc_pct]
    result = {}
    for deal, pct in zip(portfolio, alloc_pct):
        result[deal.company_name] = round(pct, 1)
    return result
