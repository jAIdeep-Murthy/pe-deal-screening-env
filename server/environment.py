"""Environment state machine for PE Deal Screening OpenEnv."""
import uuid
from typing import Any, Dict, List, Literal, Optional

from pe_env.data import (
    generate_deal,
    generate_portfolio,
    compute_ground_truth_decision,
    compute_optimal_allocation,
)
from pe_env.models import (
    DealTeaser,
    DealScreeningObservation,
    ICMemoObservation,
    PortfolioPrioritizationObservation,
    StepResult,
    ResetResponse,
)
from server.graders import grade_deal_screening, grade_ic_memo, grade_portfolio

TASKS = ["deal_screening", "ic_memo", "portfolio_prioritization"]

# In-memory episode store {episode_id: episode_state}
_episodes: Dict[str, Dict[str, Any]] = {}


def reset(task: Optional[str] = None, seed: Optional[int] = None) -> ResetResponse:
    """Create a new episode. If task is None, cycles through tasks."""
    if task is None:
        task = "deal_screening"
    if task not in TASKS:
        raise ValueError(f"Unknown task: {task}. Valid: {TASKS}")

    episode_id = str(uuid.uuid4())
    if seed is None:
        import random
        seed = random.randint(0, 99999)

    if task == "deal_screening":
        deal = generate_deal(seed)
        gt_decision = compute_ground_truth_decision(deal)
        obs = DealScreeningObservation(
            episode_id=episode_id,
            deal=deal,
        )
        _episodes[episode_id] = {
            "task": task,
            "seed": seed,
            "step": 0,
            "done": False,
            "deal": deal,
            "gt_decision": gt_decision,
            "observation": obs,
        }
        return ResetResponse(
            observation=obs.model_dump(),
            episode_id=episode_id,
            task=task,
        )

    elif task == "ic_memo":
        deal = generate_deal(seed)
        # For IC memo, we always use a deal that should be INVEST
        # Try seeds until we get an INVEST deal
        attempts = 0
        while compute_ground_truth_decision(deal) != "INVEST" and attempts < 50:
            seed += 1
            deal = generate_deal(seed)
            attempts += 1
        obs = ICMemoObservation(
            episode_id=episode_id,
            deal=deal,
        )
        _episodes[episode_id] = {
            "task": task,
            "seed": seed,
            "step": 0,
            "done": False,
            "deal": deal,
            "observation": obs,
        }
        return ResetResponse(
            observation=obs.model_dump(),
            episode_id=episode_id,
            task=task,
        )

    elif task == "portfolio_prioritization":
        portfolio = generate_portfolio(seed)
        optimal_alloc = compute_optimal_allocation(portfolio)
        obs = PortfolioPrioritizationObservation(
            episode_id=episode_id,
            portfolio=portfolio,
        )
        _episodes[episode_id] = {
            "task": task,
            "seed": seed,
            "step": 0,
            "done": False,
            "portfolio": portfolio,
            "optimal_alloc": optimal_alloc,
            "observation": obs,
        }
        return ResetResponse(
            observation=obs.model_dump(),
            episode_id=episode_id,
            task=task,
        )


def step(episode_id: str, action: Dict[str, Any]) -> StepResult:
    """Process an agent action and return reward."""
    if episode_id not in _episodes:
        raise ValueError(f"Unknown episode_id: {episode_id}")

    ep = _episodes[episode_id]
    if ep["done"]:
        return StepResult(reward=0.0, done=True, info={"error": "Episode already done"})

    task = ep["task"]
    ep["step"] += 1

    try:
        if task == "deal_screening":
            reward, info = grade_deal_screening(
                action=action,
                gt_decision=ep["gt_decision"],
                deal=ep["deal"],
            )
        elif task == "ic_memo":
            reward, info = grade_ic_memo(
                action=action,
                deal=ep["deal"],
            )
        elif task == "portfolio_prioritization":
            reward, info = grade_portfolio(
                action=action,
                portfolio=ep["portfolio"],
                optimal_alloc=ep["optimal_alloc"],
            )
        else:
            reward, info = 0.0, {"error": f"Unknown task: {task}"}
    except Exception as e:
        reward, info = 0.0, {"error": str(e), "action_received": action}

    ep["done"] = True
    return StepResult(
        observation=None,
        reward=reward,
        done=True,
        info=info,
    )


def get_episode(episode_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve episode state (for debugging)."""
    return _episodes.get(episode_id)
