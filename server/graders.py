from typing import Any, Dict, List


def grade_triage(decision: str, reason: str, scenario: Dict[str, Any]) -> float:
    """Grade Task 1: Deal Triage. Returns reward in [0.0, 1.0]."""
    ground_truth = scenario["ground_truth"]
    fit_flags = scenario["fit_flags"]
    decision = decision.strip().upper()

    # Decision correctness: 0.5 points
    decision_score = 0.5 if decision == ground_truth else 0.0
    # Partial credit: adjacent decisions get 0.25
    adjacency = {"PASS": 0, "LIGHT_DD": 1, "DEEP_DIVE": 2}
    if decision_score == 0.0 and decision in adjacency and ground_truth in adjacency:
        if abs(adjacency.get(decision, -1) - adjacency.get(ground_truth, -1)) == 1:
            decision_score = 0.25

    # Rationale quality: up to 0.5 points
    reason_lower = reason.lower()
    factor_keywords = {
        "sector": ["sector", "industry", "vertical"],
        "geography": ["geography", "region", "location", "country"],
        "ebitda": ["ebitda", "size", "earnings"],
        "margin": ["margin", "profitability", "ebitda margin"],
        "leverage": ["leverage", "debt", "net debt"],
    }
    hits = 0
    for factor, keywords in factor_keywords.items():
        if any(kw in reason_lower for kw in keywords):
            hits += 1
    rationale_score = min(hits / 5.0, 1.0) * 0.5

    return min(round(decision_score + rationale_score, 4), 1.0)


def grade_memo(memo_text: str, scenario: Dict[str, Any]) -> float:
    """Grade Task 2: IC Memo. Returns reward in [0.0, 1.0]."""
    memo_lower = memo_text.lower()
    score = 0.0
    total_items = 0

    # Section presence: 3 points (each worth 1/8 of total)
    sections = {
        "investment thesis": ["investment thesis", "## investment thesis"],
        "key risks": ["key risks", "## key risks"],
        "next steps": ["next steps", "## next steps"],
    }
    for section, variants in sections.items():
        total_items += 1
        if any(v in memo_lower for v in variants):
            score += 1.0

    # Required positives: up to 3 points
    required_positives = scenario.get("required_positives", [])
    for positive in required_positives:
        total_items += 1
        pos_words = positive.lower().split()
        if any(word in memo_lower for word in pos_words):
            score += 1.0

    # Required risks: up to 2 points
    required_risks = scenario.get("required_risks", [])
    for risk in required_risks:
        total_items += 1
        risk_words = risk.lower().split()
        if any(word in memo_lower for word in risk_words):
            score += 1.0

    # Company name mentioned
    total_items += 1
    if scenario.get("company_name", "").lower() in memo_lower:
        score += 1.0

    # Contradiction penalty: if memo says "no risk" in risk section
    if "no risk" in memo_lower or "risk free" in memo_lower:
        score = max(0.0, score - 1.0)

    if total_items == 0:
        return 0.0
    return min(round(score / total_items, 4), 1.0)


def grade_portfolio(selected_deals: List[str], allocations: Dict[str, float],
                   rationale: str, scenario: Dict[str, Any]) -> float:
    """Grade Task 3: Portfolio Prioritization. Returns reward in [0.0, 1.0]."""
    candidates = {c["deal_id"]: c for c in scenario["candidates"]}
    fund_size = scenario["fund_size_mm"]
    max_single = scenario["max_single_deal_pct"]
    sector_cap = scenario["sector_cap_pct"]
    irr_min = scenario["target_portfolio_irr_min"]
    irr_max = scenario["target_portfolio_irr_max"]
    min_deals = scenario["min_deals"]
    max_deals = scenario["max_deals"]

    # Validity check
    valid_ids = set(candidates.keys())
    selected_set = set(selected_deals)
    invalid_picks = selected_set - valid_ids
    if invalid_picks:
        return 0.0

    # Count/range check
    n_selected = len(selected_set)
    if n_selected < min_deals or n_selected > max_deals:
        return 0.1

    # Constraint satisfaction (0.4 weight)
    violations = 0
    alloc_sum = sum(allocations.values())

    # Allocation sum must be within 0.95-1.05
    if not (0.90 <= alloc_sum <= 1.10):
        violations += 1

    # Single deal cap
    for deal_id, alloc in allocations.items():
        if alloc > max_single + 0.01:
            violations += 1

    # Sector cap
    sector_allocs: Dict[str, float] = {}
    for deal_id in selected_deals:
        if deal_id not in candidates:
            continue
        sector = candidates[deal_id]["sector"]
        alloc = allocations.get(deal_id, 0.0)
        sector_allocs[sector] = sector_allocs.get(sector, 0.0) + alloc
    for sector, total_alloc in sector_allocs.items():
        if total_alloc > sector_cap + 0.01:
            violations += 1

    constraint_score = max(0.0, 1.0 - violations * 0.3) * 0.4

    # Diversification quality (0.3 weight)
    unique_sectors = len(set(candidates[d]["sector"] for d in selected_deals if d in candidates))
    max_possible_sectors = min(len(valid_ids), max_deals)
    diversification_score = min(unique_sectors / max(max_possible_sectors, 1), 1.0) * 0.3

    # Expected IRR within target band (0.3 weight)
    weighted_irr = 0.0
    for deal_id in selected_deals:
        if deal_id in candidates:
            alloc = allocations.get(deal_id, 1.0 / n_selected)
            weighted_irr += candidates[deal_id]["target_irr_pct"] * alloc
    if alloc_sum > 0:
        weighted_irr = weighted_irr / alloc_sum

    if irr_min <= weighted_irr <= irr_max:
        irr_score = 0.3
    elif weighted_irr < irr_min:
        irr_score = max(0.0, 0.3 * (weighted_irr / irr_min))
    else:
        irr_score = max(0.0, 0.3 * (irr_max / weighted_irr))

    total = constraint_score + diversification_score + irr_score
    return min(round(total, 4), 1.0)
