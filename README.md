# PE Deal Screening & IC Assistant

> **OpenEnv environment** | Meta PyTorch x Scaler Hackathon | Round 1

A realistic simulation of a **private equity associate's daily workflow**: screening inbound deal teasers, writing Investment Committee memos, and constructing a diversified portfolio under fund mandate constraints.

This environment trains and evaluates AI agents on real-world financial judgment tasks — not toy games.

---

## Environment Overview

| Property | Value |
|---|---|
| Tasks | 3 (Easy, Medium, Hard) |
| Episode type | Single-step (done=True after one action) |
| Reward range | [0.0, 1.0] per task |
| Grading | Fully deterministic, rule-based |
| Data | Synthetic but realistic (seeded, reproducible) |
| Port | 7860 |

---

## Tasks

### Task 1: Deal Triage (Easy)

**What the agent sees:** A deal teaser with company sector, geography, revenue, EBITDA, margin %, leverage, plus the fund's mandate (allowed sectors, geographies, size range, margin floor, leverage cap).

**What it must do:** Classify the deal as `PASS`, `LIGHT_DD`, or `DEEP_DIVE` and provide a rationale.

**How it is graded:**
- 0.5 pts: Correct decision (partial credit 0.25 for adjacent decision)
- 0.5 pts: Rationale mentions relevant factors (sector, geography, EBITDA, margin, leverage)

**Reward range:** [0.0, 1.0]

**Example observation:**
```json
{
  "task_id": "deal_triage",
  "company_name": "Summit Software",
  "sector": "Technology",
  "geography": "North America",
  "ebitda_mm": 42.5,
  "ebitda_margin_pct": 28.3,
  "leverage_x": 3.2,
  "mandate_sectors": ["Technology", "Healthcare", "Business Services"]
}
```

**Example action:**
```json
{"decision": "DEEP_DIVE", "reason": "Sector and geography fit mandate; EBITDA in range; strong margin above floor; leverage within cap."}
```

---

### Task 2: Mini IC Memo (Medium)

**What the agent sees:** 3-year financials (Revenue, EBITDA, FCF), market notes, risk notes, entry EV, target IRR, and a checklist of required positives and risks to address.

**What it must do:** Write a structured memo with exactly three sections: `## Investment Thesis`, `## Key Risks`, `## Next Steps`.

**How it is graded (deterministic checklist):**
- Section presence (3 items)
- Required positives mentioned (2 items)
- Required risks mentioned (2-3 items)
- Company name mentioned (1 item)
- Penalty: -1 for contradictions (e.g. "no risk")

**Reward range:** [0.0, 1.0] = items_hit / total_items

---

### Task 3: Portfolio Prioritization (Hard)

**What the agent sees:** 5-6 candidate deals with sector, EBITDA, entry EV, target IRR, risk score, plus fund constraints (max 30% per deal, max 50% per sector, target IRR 20-35%).

**What it must do:** Select 3-4 deals and allocate capital (allocations must sum to 1.0).

**How it is graded (algorithmic):**
- Constraint satisfaction: 40% of score
- Sector diversification quality: 30%
- Weighted portfolio IRR within target band: 30%
- Hard penalty: 0.0 for invalid deal IDs; 0.1 for wrong deal count

**Reward range:** [0.0, 1.0]

---

## Observation & Action Spaces

### Observation (Pydantic models)
```python
class TriageObservation(Observation):   # Task 1
    company_name: str
    sector: str
    ebitda_mm: float
    mandate_sectors: List[str]
    ...

class MemoObservation(Observation):     # Task 2
    revenue_y1_mm: float
    required_positives: List[str]
    required_risks: List[str]
    ...

class PortfolioObservation(Observation): # Task 3
    candidates: List[Dict]
    max_single_deal_pct: float
    ...
```

### Action (Pydantic models)
```python
class TriageAction(Action):       # {decision, reason}
class MemoAction(Action):         # {memo_text}
class PortfolioAction(Action):    # {selected_deals, allocations, rationale}
```

---

## Reward Shaping

All rewards provide **partial credit** — no binary pass/fail:
- Task 1: correct decision (0.5) + rationale quality (0.5)
- Task 2: fraction of checklist items hit
- Task 3: weighted sum of constraint, diversification, IRR components

---

## Baseline Scores (gemini-1.5-flash, 3 episodes each)

| Task | Avg Score |
|---|---|
| deal_triage | ~0.75 |
| ic_memo | ~0.70 |
| portfolio_prioritization | ~0.55 |
| **Overall** | **~0.67** |

---

## Setup & Running

### Environment Variables
```bash
export HF_TOKEN=your_gemini_or_openai_api_key
export API_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
export MODEL_NAME=gemini-1.5-flash
export SPACE_URL=http://localhost:7860  # or your HF Space URL
```

### Run locally with Docker
```bash
docker build -t pe-deal-screening-env .
docker run -p 7860:7860 pe-deal-screening-env
```

### Run inference script
```bash
pip install -r requirements.txt
SPACE_URL=http://localhost:7860 python inference.py
```

### OpenEnv validation
```bash
pip install openenv-core
openenv validate
```

---

## Project Structure
```
pe-deal-screening-env/
├── openenv.yaml          # OpenEnv manifest
├── inference.py          # Baseline agent (root, required)
├── Dockerfile            # Container definition
├── requirements.txt
├── README.md
├── pe_env/
│   ├── __init__.py
│   ├── models.py         # Pydantic Observation/Action/State models
│   └── data.py           # Deterministic synthetic deal generator
└── server/
    ├── __init__.py
    ├── app.py            # FastAPI app (create_fastapi_app)
    ├── environment.py    # reset() / step() / state logic
    └── graders.py        # Rule-based graders for all 3 tasks
```

---

## Hugging Face Space

Deploy as a Docker Space:
1. Create Space at huggingface.co/new-space (SDK: Docker)
2. Push this repo or link to GitHub
3. Set `HF_TOKEN` as a Space secret
4. Space URL: `https://huggingface.co/spaces/jAIdeep-Murthy/pe-deal-screening-env`

---

*Built for the Meta PyTorch x Scaler OpenEnv Hackathon — Round 1, April 2026*
