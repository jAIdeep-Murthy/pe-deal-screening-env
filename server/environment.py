import json
import sys
import uuid
from typing import Any, Optional

sys.path.insert(0, "/app")

from openenv.core.env_server import Environment
from pe_env.models import (
    TriageObservation, TriageAction,
    MemoObservation, MemoAction,
    PortfolioObservation, PortfolioAction,
    DealState,
)
from pe_env.data import make_triage_scenario, make_memo_scenario, make_portfolio_scenario
from server.graders import grade_triage, grade_memo, grade_portfolio

TASKS = ["deal_triage", "ic_memo", "portfolio_prioritization"]


class PEDealScreeningEnv(Environment):
    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self):
        self._state = DealState()
        self._scenario: dict = {}
        self._task_index: int = 0
        self._seed: int = 42
        self._stepped: bool = False

    def reset(self, seed=None, episode_id=None, task=None, **kwargs):
        self._stepped = False
        if seed is not None:
            self._seed = int(seed)
        if task is not None and task in TASKS:
            self._task_index = TASKS.index(task)
        task_id = TASKS[self._task_index % len(TASKS)]
        self._state = DealState(
            episode_id=episode_id or str(uuid.uuid4()),
            step_count=0,
            task_id=task_id,
            seed=self._seed,
        )
        if task_id == "deal_triage":
            self._scenario = make_triage_scenario(self._seed)
            self._state.ground_truth = self._scenario["ground_truth"]
            obs_data = {k: v for k, v in self._scenario.items()
                        if k not in ("ground_truth", "fit_flags")}
            return TriageObservation(done=False, reward=None, **obs_data)
        elif task_id == "ic_memo":
            self._scenario = make_memo_scenario(self._seed)
            self._state.checklist_items = (
                3 + len(self._scenario["required_positives"]) +
                len(self._scenario["required_risks"]) + 1
            )
            return MemoObservation(done=False, reward=None, **self._scenario)
        else:
            self._scenario = make_portfolio_scenario(self._seed)
            return PortfolioObservation(done=False, reward=None, **self._scenario)

    def step(self, action, timeout_s=None, **kwargs):
        self._state.step_count += 1
        self._stepped = True
        task_id = self._state.task_id

        if task_id == "deal_triage":
            if isinstance(action, dict):
                decision = action.get("decision", "PASS")
                reason = action.get("reason", "")
            else:
                decision = getattr(action, "decision", "PASS")
                reason = getattr(action, "reason", "")
            # Try to parse JSON from free-text
            if isinstance(decision, str) and decision.startswith("{"):
                try:
                    parsed = json.loads(decision)
                    decision = parsed.get("decision", "PASS")
                    reason = parsed.get("reason", reason)
                except Exception:
                    pass
            reward = grade_triage(str(decision), str(reason), self._scenario)
            obs = TriageObservation(
                done=True, reward=reward,
                **{k: v for k, v in self._scenario.items()
                   if k not in ("ground_truth", "fit_flags")}
            )
            return obs, reward, True, {"task_id": task_id, "ground_truth": self._scenario["ground_truth"]}

        elif task_id == "ic_memo":
            if isinstance(action, dict):
                memo_text = action.get("memo_text", str(action))
            else:
                memo_text = getattr(action, "memo_text", str(action))
            reward = grade_memo(str(memo_text), self._scenario)
            obs = MemoObservation(done=True, reward=reward, **self._scenario)
            return obs, reward, True, {"task_id": task_id}

        else:
            if isinstance(action, dict):
                selected = action.get("selected_deals", [])
                allocations = action.get("allocations", {})
                rationale = action.get("rationale", "")
            else:
                selected = getattr(action, "selected_deals", [])
                allocations = getattr(action, "allocations", {})
                rationale = getattr(action, "rationale", "")
            # Parse from JSON string if needed
            if isinstance(selected, str):
                try:
                    parsed = json.loads(selected)
                    selected = parsed.get("selected_deals", [])
                    allocations = parsed.get("allocations", {})
                    rationale = parsed.get("rationale", "")
                except Exception:
                    selected = []
            reward = grade_portfolio(selected, allocations, str(rationale), self._scenario)
            obs = PortfolioObservation(done=True, reward=reward, **self._scenario)
            return obs, reward, True, {"task_id": task_id}

    @property
    def state(self) -> DealState:
        return self._state
