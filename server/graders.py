"""Rule-based graders for all 3 PE Deal Screening tasks."""
from typing import Any, Dict, List, Tuple

from pe_env.models import DealTeaser, PortfolioDeal


def grade_deal_screening(
    action: Dict[str, Any],
    gt_decision: str,
    deal: DealTeaser,
) -> Tuple[float, Dict[str, Any]]:
    """Grade Task 1: Deal Screening. Returns (reward, info) where reward in (0.001, 0.999)."""
    decision = str(action.get("decision", "")).strip().upper()
    confidence = float(action.get("confidence", 0.5))
    rationale = str(action.get("rationale", ""))

    info = {
        "gt_decision": gt_decision,
        "agent_decision": decision,
        "confidence": confidence,
        "breakdown": {},
    }

    # 1. Decision correctness (0.6 weight)
    decision_correct = decision == gt_decision
    decision_score = 0.6 if decision_correct else 0.0
    info["breakdown"]["decision_correct"] = decision_correct
    info["breakdown"]["decision_score"] = decision_score

    # 2. Confidence calibration (0.2 weight)
    if decision_correct:
        calib_score = 0.2 * confidence
    else:
        calib_score = 0.2 * (1.0 - confidence)
    info["breakdown"]["calibration_score"] = round(calib_score, 3)

    # 3. Rationale quality (0.2 weight)
    rationale_lower = rationale.lower()
    keywords = [
        "ebitda", "margin", "revenue", "growth", "multiple", "debt",
        "leverage", "valuation", "sector", "risk", "cash flow",
        deal.sector.lower(), deal.hq_country.lower(),
    ]
    keyword_hits = sum(1 for kw in keywords if kw in rationale_lower)
    rationale_length_ok = len(rationale) >= 50
    rationale_score = 0.0
    if rationale_length_ok:
        rationale_score += 0.1
    if keyword_hits >= 3:
        rationale_score += 0.1
    info["breakdown"]["rationale_keyword_hits"] = keyword_hits
    info["breakdown"]["rationale_score"] = round(rationale_score, 3)

    raw_total = round(decision_score + calib_score + rationale_score, 4)
    total = float(max(0.001, min(0.999, raw_total)))
    info["total_reward"] = total
    return total, info


def grade_ic_memo(
    action: Dict[str, Any],
    deal: DealTeaser,
) -> Tuple[float, Dict[str, Any]]:
    """Grade Task 2: IC Memo Writing. Returns (reward, info) where reward in (0.001, 0.999)."""
    exec_summary = str(action.get("executive_summary", ""))
    thesis = str(action.get("investment_thesis", ""))
    risks = str(action.get("key_risks", ""))
    financials = str(action.get("financial_highlights", ""))
    recommendation = str(action.get("recommendation", "")).strip().upper()

    info = {"breakdown": {}}

    # 1. Sections present (0.3 weight)
    sections = {
        "executive_summary": len(exec_summary) >= 50,
        "investment_thesis": len(thesis) >= 50,
        "key_risks": len(risks) >= 30,
        "financial_highlights": len(financials) >= 30,
        "recommendation": recommendation in ["INVEST", "PASS", "CONDITIONAL"],
    }
    sections_score = 0.3 * (sum(sections.values()) / len(sections))
    info["breakdown"]["sections"] = sections
    info["breakdown"]["sections_score"] = round(sections_score, 3)

    # 2. Thesis coherence (0.3 weight)
    thesis_lower = thesis.lower()
    thesis_keywords = ["growth", "margin", "market", "competitive", "return", "value", "ebitda", "multiple"]
    thesis_hits = sum(1 for kw in thesis_keywords if kw in thesis_lower)
    thesis_score = min(0.3, 0.3 * thesis_hits / 4)
    info["breakdown"]["thesis_keyword_hits"] = thesis_hits
    info["breakdown"]["thesis_score"] = round(thesis_score, 3)

    # 3. Risk identification (0.2 weight)
    risks_lower = risks.lower()
    risk_keywords = ["risk", "concern", "challenge", "competition", "leverage", "macro", "regulatory", "execution"]
    risk_hits = sum(1 for kw in risk_keywords if kw in risks_lower)
    risk_score = min(0.2, 0.2 * risk_hits / 3)
    info["breakdown"]["risk_keyword_hits"] = risk_hits
    info["breakdown"]["risk_score"] = round(risk_score, 3)

    # 4. Recommendation alignment (0.2 weight)
    rec_score = 0.2 if recommendation == "INVEST" else (0.1 if recommendation == "CONDITIONAL" else 0.0)
    info["breakdown"]["recommendation"] = recommendation
    info["breakdown"]["recommendation_score"] = rec_score

    raw_total = round(sections_score + thesis_score + risk_score + rec_score, 4)
    total = float(max(0.001, min(0.999, raw_total)))
    info["total_reward"] = total
    return total, info


def grade_portfolio(
    action: Dict[str, Any],
    portfolio: List[PortfolioDeal],
    optimal_alloc: Dict[str, float],
) -> Tuple[float, Dict[str, Any]]:
    """Grade Task 3: Portfolio Prioritization. Returns (reward, info) where reward in (0.001, 0.999)."""
    allocations = action.get("allocations", [])
    rationale = str(action.get("rationale", ""))
    info = {"breakdown": {}}

    if not allocations:
        return 0.001, {"error": "No allocations provided", "total_reward": 0.001}

    # Build allocation map
    alloc_map: Dict[str, float] = {}
    for a in allocations:
        name = str(a.get("company_name", ""))
        pct = float(a.get("allocation_pct", 0.0))
        alloc_map[name] = pct

    portfolio_names = {d.company_name for d in portfolio}

    # 1. Constraint satisfaction (0.4 weight)
    constraint_violations = 0
    total_pct = sum(alloc_map.values())

    if abs(total_pct - 100.0) > 5.0:
        constraint_violations += 1

    for name, pct in alloc_map.items():
        if pct > 42.0:
            constraint_violations += 1

    for name, pct in alloc_map.items():
        if 0 < pct < 8.0:
            constraint_violations += 1

    max_violations = len(portfolio) + 2
    constraint_score = 0.4 * max(0, 1 - constraint_violations / max_violations)
    info["breakdown"]["total_pct"] = round(total_pct, 1)
    info["breakdown"]["constraint_violations"] = constraint_violations
    info["breakdown"]["constraint_score"] = round(constraint_score, 3)

    # 2. Return optimization (0.4 weight)
    agent_irr = 0.0
    optimal_irr = 0.0
    for deal in portfolio:
        agent_w = alloc_map.get(deal.company_name, 0.0) / 100.0
        optimal_w = optimal_alloc.get(deal.company_name, 0.0) / 100.0
        agent_irr += agent_w * deal.expected_irr_pct
        optimal_irr += optimal_w * deal.expected_irr_pct

    if optimal_irr > 0:
        return_score = 0.4 * min(1.0, agent_irr / optimal_irr)
    else:
        return_score = 0.2
    info["breakdown"]["agent_weighted_irr"] = round(agent_irr, 2)
    info["breakdown"]["optimal_weighted_irr"] = round(optimal_irr, 2)
    info["breakdown"]["return_score"] = round(return_score, 3)

    # 3. Rationale quality (0.2 weight)
    rationale_lower = rationale.lower()
    rat_keywords = ["irr", "risk", "return", "allocation", "diversif", "sector", "weight", "portfolio"]
    rat_hits = sum(1 for kw in rat_keywords if kw in rationale_lower)
    rationale_score = min(0.2, 0.2 * rat_hits / 4)
    info["breakdown"]["rationale_keyword_hits"] = rat_hits
    info["breakdown"]["rationale_score"] = round(rationale_score, 3)

    raw_total = round(constraint_score + return_score + rationale_score, 4)
    total = float(max(0.001, min(0.999, raw_total)))
    info["total_reward"] = total
    return total, info
