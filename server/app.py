"""FastAPI application for PE Deal Screening OpenEnv."""
import sys
sys.path.insert(0, "/app")

from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import server.environment as env
from pe_env.models import StepResult, ResetResponse

app = FastAPI(
    title="PE Deal Screening & IC Assistant",
    description="OpenEnv environment for PE deal screening, IC memo writing & portfolio prioritization.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ResetRequest(BaseModel):
    task: Optional[str] = None  # "deal_screening", "ic_memo", "portfolio_prioritization"
    seed: Optional[int] = None


class StepRequest(BaseModel):
    episode_id: str
    action: Dict[str, Any]


@app.get("/health")
def health():
    """Health check endpoint."""
    return {"status": "ok", "service": "pe-deal-screening-env"}


@app.get("/info")
def info():
    """Environment info."""
    return {
        "name": "pe_deal_screening_env",
        "version": "1.0.0",
        "tasks": env.TASKS,
        "description": "PE deal screening, IC memo writing, and portfolio prioritization.",
        "endpoints": ["/reset", "/step", "/health", "/info"],
    }


@app.post("/reset", response_model=ResetResponse)
def reset(request: ResetRequest):
    """Reset/start a new episode."""
    try:
        result = env.reset(task=request.task, seed=request.seed)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")


@app.post("/step", response_model=StepResult)
def step(request: StepRequest):
    """Submit an action and get reward."""
    try:
        result = env.step(episode_id=request.episode_id, action=request.action)
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")


@app.get("/episode/{episode_id}")
def get_episode(episode_id: str):
    """Get episode state (debug endpoint)."""
    ep = env.get_episode(episode_id)
    if ep is None:
        raise HTTPException(status_code=404, detail=f"Episode {episode_id} not found")
    # Serialize safely
    return {
        "episode_id": episode_id,
        "task": ep["task"],
        "step": ep["step"],
        "done": ep["done"],
        "seed": ep["seed"],
    }
