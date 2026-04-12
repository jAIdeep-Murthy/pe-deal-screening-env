"""inference.py - PE Deal Screening & IC Assistant baseline agent.
Uses OpenAI API via env vars (API_BASE_URL, MODEL_NAME, HF_TOKEN).
Emits [START], [STEP], [END] logs per hackathon spec.
"""
import json
import os
import sys
from typing import Any, Dict, List, Optional

import requests
from openai import OpenAI


def log_start(task: str, env: str, model: str):
    print(json.dumps({"tag": "[START]", "task": task, "env": env, "model": model}), flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error=None):
    print(json.dumps({"tag": "[STEP]", "step": step, "action": action, "reward": reward, "done": done, "error": error}), flush=True)


def log_end(success: bool, steps: int, score: float, rewards: list):
    print(json.dumps({"tag": "[END]", "success": success, "steps": steps, "score": score, "rewards": rewards}), flush=True)

# ---------------------------------------------------------------------------
# Configuration from environment variables (no hardcoded keys/URLs)
# ---------------------------------------------------------------------------
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o-mini")
API_KEY = os.environ.get("HF_TOKEN") or os.environ.get("OPENAI_API_KEY", "")
SPACE_URL = os.environ.get("SPACE_URL", "http://localhost:7860")

BENCHMARK = "pe_deal_screening_env"
TEMPERATURE = 0.2
MAX_TOKENS = 1024
EPISODES_PER_TASK = 3

TASKS = [
    {"name": "deal_screening",         "max_steps": 1, "max_reward": 1},
    {"name": "ic_memo",                 "max_steps": 1, "max_reward": 1},
    {"name": "portfolio_prioritization", "max_steps": 1, "max_reward": 1},
]

SYSTEM_PROMPT = """You are an expert private equity associate. You will be given deal screening scenarios.
Always respond in the exact JSON format requested. Be analytical, precise, and professional."""


def get_client() -> OpenAI:
    return OpenAI(api_key=API_KEY, base_url=API_BASE_URL)


def call_llm(client: OpenAI, messages: List[Dict]) -> str:
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
    )
    return response.choices[0].message.content.strip()


def build_deal_screening_prompt(obs: Dict) -> str:
    deal = obs["deal"]
    return f"""You are screening an inbound deal. Analyze the following deal teaser and decide INVEST or PASS.

Company: {deal['company_name']}
Sector: {deal['sector']}
HQ: {deal['hq_country']}
Founded: {deal['founded_year']}
Revenue: ${deal['revenue_mm']}M
EBITDA: ${deal['ebitda_mm']}M ({deal['ebitda_margin_pct']}% margin)
Revenue Growth: {deal['revenue_growth_pct']}% YoY
Asking EV/EBITDA: {deal['asking_ev_ebitda']}x
Net Debt/EBITDA: {deal['net_debt_ebitda']}x

Respond with ONLY valid JSON:
{{"decision": "INVEST" or "PASS", "confidence": 0-1, "rationale": "your reasoning mentioning key metrics"}}"""


def build_ic_memo_prompt(obs: Dict) -> str:
    deal = obs["deal"]
    context = deal.get("deal_context", "")
    return f"""Write an Investment Committee memo for the following deal.

Company: {deal['company_name']} | Sector: {deal['sector']} | HQ: {deal['hq_country']}
Revenue: ${deal['revenue_mm']}M | EBITDA: ${deal['ebitda_mm']}M ({deal['ebitda_margin_pct']}% margin)
Growth: {deal['revenue_growth_pct']}% | EV/EBITDA: {deal['asking_ev_ebitda']}x | Net Debt/EBITDA: {deal['net_debt_ebitda']}x
Context: {context}

Respond with ONLY valid JSON:
{{
  "executive_summary": "...",
  "investment_thesis": "mention growth, margin, market, competitive position, value creation",
  "key_risks": "identify leverage, competitive, regulatory, execution risks",
  "financial_highlights": "key financial metrics and what they imply",
  "recommendation": "INVEST" or "PASS" or "CONDITIONAL"
}}"""


def build_portfolio_prompt(obs: Dict) -> str:
    portfolio = obs["portfolio"]
    deals_str = ""
    for i, d in enumerate(portfolio, 1):
        deals_str += f"{i}. {d['company_name']} ({d['sector']}): IRR={d['expected_irr_pct']}%, Risk={d['risk_score']:.2f}, EV/EBITDA={d['asking_ev_ebitda']}x\n"
    return f"""Allocate a $100M fund across these 5 deals.
Constraints: max 40% in any single deal, min 10% if included. Maximize risk-adjusted returns.

{deals_str}
Respond with ONLY valid JSON:
{{
  "allocations": [
    {{"company_name": "...", "allocation_mm": 20, "allocation_pct": 20, "rationale": "..."}},
    ...
  ],
  "total_deployed_mm": 100,
  "rationale": "overall portfolio rationale mentioning IRR, risk, diversification, sector allocation"
}}"""


def parse_json_action(content: str) -> Dict:
    """Robustly extract JSON from LLM response."""
    # Try direct parse
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass
    # Try to find JSON block
    import re
    match = re.search(r'\{.*\}', content, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return {"error": "Failed to parse JSON", "raw": content[:200]}


def clamp_score(value: float) -> float:
    """Ensure score is strictly between 0 and 1 (not 0.0 and not 1.0)."""
    return float(max(0.001, min(0.999, value)))


def run_episode(client: OpenAI, task_name: str, episode_num: int) -> Dict:
    """Run a single episode for a given task."""
    # Reset
    reset_resp = requests.post(f"{SPACE_URL}/reset", json={"task": task_name})
    reset_data = reset_resp.json()
    episode_id = reset_data["episode_id"]
    obs = reset_data["observation"]

    log_start(task_name, BENCHMARK, MODEL_NAME)

    # Build prompt based on task
    if task_name == "deal_screening":
        prompt = build_deal_screening_prompt(obs)
    elif task_name == "ic_memo":
        prompt = build_ic_memo_prompt(obs)
    elif task_name == "portfolio_prioritization":
        prompt = build_portfolio_prompt(obs)
    else:
        prompt = f"Task: {task_name}\nObservation: {json.dumps(obs)}"

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]

    # Call LLM
    llm_response = call_llm(client, messages)
    action = parse_json_action(llm_response)

    # Step
    step_resp = requests.post(f"{SPACE_URL}/step", json={"episode_id": episode_id, "action": action})
    step_data = step_resp.json()
    reward = clamp_score(step_data["reward"])
    done = step_data["done"]
    info = step_data.get("info", {})

    log_step(1, str(action), reward, done)
    log_end(done, 1, reward, [reward])

    return {"episode_id": episode_id, "reward": reward, "info": info}


def main():
    client = get_client()
    all_results = []
    total_reward = 0
    total_episodes = 0

    for task_cfg in TASKS:
        task_name = task_cfg["name"]
        for ep in range(1, EPISODES_PER_TASK + 1):
            try:
                result = run_episode(client, task_name, ep)
                all_results.append({"task": task_name, "episode": ep, **result})
                total_reward += result["reward"]
                total_episodes += 1
            except Exception as e:
                print(f"[ERROR] task={task_name} episode={ep} error={e}")
                sys.stdout.flush()

    avg_reward = total_reward / max(total_episodes, 1)
    print(f"\n=== BENCHMARK COMPLETE ===")
    print(f"Total episodes: {total_episodes}")
    print(f"Average reward: {avg_reward:.4f}")
    print(f"Total reward: {total_reward:.4f}")


if __name__ == "__main__":
    main()
