"""inference.py - PE Deal Screening & IC Assistant baseline agent.
Uses OpenAI client (compatible with Gemini, OpenAI, etc.) via env vars.
Emits [START], [STEP], [END] logs per hackathon spec.
"""
import asyncio
import json
import os
import sys
from typing import Any, Dict, List, Optional

from openai import OpenAI

# ---------------------------------------------------------------------------
# Configuration from environment variables (no hardcoded keys/URLs)
# ---------------------------------------------------------------------------
API_BASE_URL = os.environ.get("API_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")
MODEL_NAME = os.environ.get("MODEL_NAME", "gemini-1.5-flash")
API_KEY = os.environ.get("HF_TOKEN") or os.environ.get("OPENAI_API_KEY", "")
SPACE_URL = os.environ.get("SPACE_URL", "http://localhost:7860")

BENCHMARK = "pe_deal_screening_env"
TEMPERATURE = 0.2
MAX_TOKENS = 1024
EPISODES_PER_TASK = 3

TASKS = [
    {"name": "deal_triage",              "max_steps": 1, "max_reward": 1.0},
    {"name": "ic_memo",                  "max_steps": 1, "max_reward": 1.0},
    {"name": "portfolio_prioritization", "max_steps": 1, "max_reward": 1.0},
]

SYSTEM_PROMPT = """You are an expert private equity associate. You will be given deal screening and investment committee tasks.
Always respond in the exact JSON format requested. Be analytical, precise, and professional."""


# ---------------------------------------------------------------------------
# Log helpers - strict [START], [STEP], [END] format
# ---------------------------------------------------------------------------
def log_start(task: str, env: str, model: str) -> None:
    print(json.dumps({"event": "START", "task": task, "env": env, "model": model}), flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    print(json.dumps({"event": "STEP", "step": step, "action": action[:200],
                      "reward": reward, "done": done, "error": error}), flush=True)


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    print(json.dumps({"event": "END", "success": success, "steps": steps,
                      "score": score, "rewards": rewards}), flush=True)


# ---------------------------------------------------------------------------
# HTTP helpers to call the OpenEnv server
# ---------------------------------------------------------------------------
import httpx


def env_reset(task: str, seed: int = 42) -> Dict[str, Any]:
    url = f"{SPACE_URL}/reset"
    r = httpx.post(url, json={"task": task, "seed": seed}, timeout=30)
    r.raise_for_status()
    return r.json()


def env_step(action_payload: Dict[str, Any]) -> Dict[str, Any]:
    url = f"{SPACE_URL}/step"
    r = httpx.post(url, json={"action": action_payload}, timeout=60)
    r.raise_for_status()
    return r.json()


# ---------------------------------------------------------------------------
# Prompt builders per task
# ---------------------------------------------------------------------------
def build_triage_prompt(obs: Dict[str, Any]) -> str:
    return f"""You are screening a deal for your fund.

Company: {obs.get('company_name')} | Sector: {obs.get('sector')} | Geography: {obs.get('geography')}
Revenue: ${obs.get('revenue_mm')}M | EBITDA: ${obs.get('ebitda_mm')}M | Margin: {obs.get('ebitda_margin_pct')}% | Leverage: {obs.get('leverage_x')}x

Fund Mandate:
- Sectors: {obs.get('mandate_sectors')}
- Geographies: {obs.get('mandate_geographies')}
- EBITDA range: ${obs.get('mandate_ebitda_min_mm')}M - ${obs.get('mandate_ebitda_max_mm')}M
- Min EBITDA margin: {obs.get('mandate_margin_min_pct')}%
- Max leverage: {obs.get('mandate_leverage_max_x')}x

Evaluate sector fit, geography fit, EBITDA size, margin quality, and leverage. Respond ONLY with this JSON:
{{"decision": "PASS", "reason": "..."}}
Where decision is one of: PASS, LIGHT_DD, DEEP_DIVE"""


def build_memo_prompt(obs: Dict[str, Any]) -> str:
    return f"""Write an IC memo for this investment opportunity.

Company: {obs.get('company_name')} ({obs.get('sector')}, {obs.get('geography')})
Financials (Y1/Y2/Y3):
  Revenue: ${obs.get('revenue_y1_mm')}M / ${obs.get('revenue_y2_mm')}M / ${obs.get('revenue_y3_mm')}M
  EBITDA:  ${obs.get('ebitda_y1_mm')}M / ${obs.get('ebitda_y2_mm')}M / ${obs.get('ebitda_y3_mm')}M
  FCF Y3:  ${obs.get('fcf_y3_mm')}M
Entry EV: ${obs.get('entry_ev_mm')}M | Target IRR: {obs.get('target_irr_pct')}%
Market: {obs.get('market_notes')}
Risks: {obs.get('risk_notes')}
Required positives to address: {obs.get('required_positives')}
Required risks to address: {obs.get('required_risks')}

Write the memo with EXACTLY these three sections (use these exact headings):
## Investment Thesis
## Key Risks
## Next Steps

Be specific, mention the company name, address all required positives and risks."""


def build_portfolio_prompt(obs: Dict[str, Any]) -> str:
    candidates_str = json.dumps(obs.get('candidates', []), indent=2)
    return f"""Portfolio construction task. Select the best deals for your fund.

Fund size: ${obs.get('fund_size_mm')}M
Constraints:
- Max single deal allocation: {obs.get('max_single_deal_pct')*100:.0f}%
- Max sector concentration: {obs.get('sector_cap_pct')*100:.0f}%
- Target portfolio IRR: {obs.get('target_portfolio_irr_min')}% - {obs.get('target_portfolio_irr_max')}%
- Select {obs.get('min_deals')}-{obs.get('max_deals')} deals

Candidates:
{candidates_str}

Respond ONLY with this JSON:
{{"selected_deals": ["D01", "D02", "D03"], "allocations": {{"D01": 0.30, "D02": 0.35, "D03": 0.35}}, "rationale": "Brief rationale"}}
Allocations must sum to 1.0. Respect all constraints."""


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------
def call_llm(client: OpenAI, prompt: str) -> str:
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        print(f"[DEBUG] LLM call failed: {e}", flush=True)
        return "{}"


# ---------------------------------------------------------------------------
# Parse LLM output to action payload
# ---------------------------------------------------------------------------
def parse_action(task_name: str, llm_output: str) -> Dict[str, Any]:
    # Extract JSON block
    text = llm_output.strip()
    if "```" in text:
        parts = text.split("```")
        for part in parts:
            part = part.strip().lstrip("json").strip()
            if part.startswith("{"):
                text = part
                break
    # Find first JSON object
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1:
        text = text[start:end+1]
    try:
        parsed = json.loads(text)
    except Exception:
        parsed = {}

    if task_name == "deal_triage":
        return {
            "decision": parsed.get("decision", "PASS"),
            "reason": parsed.get("reason", llm_output[:500]),
        }
    elif task_name == "ic_memo":
        return {"memo_text": llm_output}
    else:
        return {
            "selected_deals": parsed.get("selected_deals", []),
            "allocations": parsed.get("allocations", {}),
            "rationale": parsed.get("rationale", ""),
        }


# ---------------------------------------------------------------------------
# Run one episode for one task
# ---------------------------------------------------------------------------
def run_episode(client: OpenAI, task_name: str, seed: int) -> float:
    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    try:
        obs_raw = env_reset(task=task_name, seed=seed)
        obs = obs_raw.get("observation", obs_raw)

        if obs.get("done", False):
            log_end(success=False, steps=0, score=0.0, rewards=[])
            return 0.0

        # Build prompt based on task
        if task_name == "deal_triage":
            prompt = build_triage_prompt(obs)
        elif task_name == "ic_memo":
            prompt = build_memo_prompt(obs)
        else:
            prompt = build_portfolio_prompt(obs)

        llm_output = call_llm(client, prompt)
        action_payload = parse_action(task_name, llm_output)

        step_result = env_step(action_payload)
        reward = float(step_result.get("reward") or 0.0)
        done = step_result.get("done", True)
        rewards.append(reward)
        steps_taken = 1

        log_step(step=1, action=json.dumps(action_payload), reward=reward, done=done, error=None)

        score = min(max(reward, 0.0), 1.0)
        success = score >= 0.5

    except Exception as e:
        print(f"[DEBUG] Episode error: {e}", flush=True)
        log_step(step=steps_taken+1, action="", reward=0.0, done=True, error=str(e))

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    if not API_KEY:
        print("[ERROR] No API key found. Set HF_TOKEN or OPENAI_API_KEY.", flush=True)
        sys.exit(1)

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    all_scores: Dict[str, List[float]] = {}

    for task_cfg in TASKS:
        task_name = task_cfg["name"]
        task_scores = []
        for ep in range(EPISODES_PER_TASK):
            seed = 42 + ep * 17
            score = run_episode(client, task_name, seed)
            task_scores.append(score)
            print(f"[DEBUG] Task={task_name} episode={ep+1} score={score:.4f}", flush=True)
        all_scores[task_name] = task_scores

    # Summary
    print("\n=== BASELINE SCORES ===", flush=True)
    for task_name, scores in all_scores.items():
        avg = sum(scores) / len(scores) if scores else 0.0
        print(f"  {task_name}: avg={avg:.4f}  episodes={scores}", flush=True)
    overall = sum(s for scores in all_scores.values() for s in scores)
    total = sum(len(v) for v in all_scores.values())
    print(f"  OVERALL AVG: {overall/total:.4f}", flush=True)


if __name__ == "__main__":
    main()
