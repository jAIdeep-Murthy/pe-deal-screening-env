import random
from typing import Any, Dict, List

SECTORS = ["Healthcare", "Technology", "Business Services", "Industrials",
           "Consumer", "Financial Services", "Energy", "Education"]

GEOGRAPHIES = ["North America", "Western Europe", "DACH", "UK & Ireland",
               "Nordics", "Southern Europe", "Southeast Asia"]

COMPANY_NAMES = [
    "Apex Analytics", "Bridgepoint Solutions", "Crestview Systems",
    "Delta MedTech", "Evergreen Software", "Falcon Industrials",
    "Granite Business Services", "Harbour Capital", "Ironclad Tech",
    "Junction Healthcare", "Keystone Education", "Landmark Consumer",
    "Meridian Financial", "Northstar Energy", "Oaktree Logistics",
    "Pinnacle Business Services", "Quantum Technology", "Ridgeline Healthcare",
    "Summit Software", "Thornberry Industrials",
]

MARKET_NOTES = [
    "Fragmented market with 200+ competitors; top-3 hold ~30% share. Secular tailwind from digital transformation. TAM growing ~12% p.a.",
    "Consolidated market dominated by 2 incumbents. Strong pricing power. Regulatory moat. Recurring revenue >80%.",
    "Niche B2B market with sticky customer relationships. Low churn (<5% p.a.), long contract durations (avg 4.2 years).",
]


def make_triage_scenario(seed: int) -> Dict[str, Any]:
    rng = random.Random(seed)
    sector = rng.choice(SECTORS)
    geography = rng.choice(GEOGRAPHIES)
    name = rng.choice(COMPANY_NAMES)
    ebitda_mm = round(rng.uniform(5, 120), 1)
    revenue_mm = round(ebitda_mm / rng.uniform(0.10, 0.45), 1)
    ebitda_margin_pct = round((ebitda_mm / revenue_mm) * 100, 1)
    leverage_x = round(rng.uniform(1.0, 6.5), 1)
    mandate_sectors = rng.sample(SECTORS, k=3)
    if rng.random() < 0.6 and sector not in mandate_sectors:
        mandate_sectors[0] = sector
    mandate_geos = rng.sample(GEOGRAPHIES, k=3)
    if rng.random() < 0.6 and geography not in mandate_geos:
        mandate_geos[0] = geography
    mandate_ebitda_min = round(rng.uniform(5, 30), 0)
    mandate_ebitda_max = round(mandate_ebitda_min + rng.uniform(20, 90), 0)
    mandate_margin_min = round(rng.uniform(8, 20), 1)
    mandate_leverage_max = round(rng.uniform(3.5, 5.5), 1)
    sector_fit = sector in mandate_sectors
    geo_fit = geography in mandate_geos
    ebitda_fit = mandate_ebitda_min <= ebitda_mm <= mandate_ebitda_max
    margin_fit = ebitda_margin_pct >= mandate_margin_min
    leverage_fit = leverage_x <= mandate_leverage_max
    fit_count = sum([sector_fit, geo_fit, ebitda_fit, margin_fit, leverage_fit])
    if not sector_fit or not geo_fit:
        ground_truth = "PASS"
    elif fit_count == 5:
        ground_truth = "DEEP_DIVE"
    elif fit_count >= 3:
        ground_truth = "LIGHT_DD"
    else:
        ground_truth = "PASS"
    return {
        "company_name": name, "sector": sector, "geography": geography,
        "revenue_mm": revenue_mm, "ebitda_mm": ebitda_mm,
        "ebitda_margin_pct": ebitda_margin_pct, "leverage_x": leverage_x,
        "mandate_sectors": mandate_sectors, "mandate_geographies": mandate_geos,
        "mandate_ebitda_min_mm": mandate_ebitda_min, "mandate_ebitda_max_mm": mandate_ebitda_max,
        "mandate_margin_min_pct": mandate_margin_min, "mandate_leverage_max_x": mandate_leverage_max,
        "ground_truth": ground_truth,
        "fit_flags": {"sector_fit": sector_fit, "geo_fit": geo_fit,
                      "ebitda_fit": ebitda_fit, "margin_fit": margin_fit, "leverage_fit": leverage_fit},
    }


def make_memo_scenario(seed: int) -> Dict[str, Any]:
    rng = random.Random(seed + 1000)
    sector = rng.choice(SECTORS)
    geography = rng.choice(GEOGRAPHIES)
    name = rng.choice(COMPANY_NAMES)
    ebitda_y1 = round(rng.uniform(15, 60), 1)
    growth = rng.uniform(0.08, 0.25)
    ebitda_y2 = round(ebitda_y1 * (1 + growth), 1)
    ebitda_y3 = round(ebitda_y2 * (1 + growth * rng.uniform(0.7, 1.2)), 1)
    margin_y1 = rng.uniform(0.15, 0.35)
    revenue_y1 = round(ebitda_y1 / margin_y1, 1)
    revenue_y2 = round(ebitda_y2 / (margin_y1 + rng.uniform(0, 0.03)), 1)
    revenue_y3 = round(ebitda_y3 / (margin_y1 + rng.uniform(0, 0.05)), 1)
    fcf_y3 = round(ebitda_y3 * rng.uniform(0.65, 0.90), 1)
    entry_ev = round(ebitda_y3 * rng.uniform(8, 14), 1)
    target_irr = round(rng.uniform(18, 30), 1)
    market_note = rng.choice(MARKET_NOTES)
    risk_names = ["customer concentration", "key-man risk", "leverage", "technology disruption", "macro sensitivity"]
    rng.shuffle(risk_names)
    chosen_risks = risk_names[:rng.randint(2, 3)]
    positives = ["revenue growth", "margin expansion", "strong FCF conversion", "recurring revenue model"]
    rng.shuffle(positives)
    chosen_positives = positives[:2]
    risk_note = "; ".join([f"{r} risk present" for r in chosen_risks])
    return {
        "company_name": name, "sector": sector, "geography": geography,
        "revenue_y1_mm": revenue_y1, "revenue_y2_mm": revenue_y2, "revenue_y3_mm": revenue_y3,
        "ebitda_y1_mm": ebitda_y1, "ebitda_y2_mm": ebitda_y2, "ebitda_y3_mm": ebitda_y3,
        "fcf_y3_mm": fcf_y3, "market_notes": market_note, "risk_notes": risk_note,
        "entry_ev_mm": entry_ev, "target_irr_pct": target_irr,
        "required_positives": chosen_positives, "required_risks": chosen_risks,
    }


def make_portfolio_scenario(seed: int) -> Dict[str, Any]:
    rng = random.Random(seed + 2000)
    n = rng.randint(5, 6)
    candidates = []
    for i in range(n):
        s = rng.choice(SECTORS)
        ebitda = round(rng.uniform(10, 80), 1)
        ev = round(ebitda * rng.uniform(7, 13), 1)
        irr = round(rng.uniform(15, 40), 1)
        risk = round(rng.uniform(1.0, 5.0), 1)
        candidates.append({"deal_id": f"D{i+1:02d}", "company_name": rng.choice(COMPANY_NAMES),
                            "sector": s, "ebitda_mm": ebitda, "entry_ev_mm": ev,
                            "target_irr_pct": irr, "risk_score": risk})
    fund_size = round(rng.uniform(200, 600), 0)
    return {
        "candidates": candidates, "fund_size_mm": fund_size,
        "max_single_deal_pct": 0.30, "sector_cap_pct": 0.50,
        "target_portfolio_irr_min": 20.0, "target_portfolio_irr_max": 35.0,
        "min_deals": 3, "max_deals": 4,
    }
