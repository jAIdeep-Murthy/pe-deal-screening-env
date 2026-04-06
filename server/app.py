"""FastAPI application for PE Deal Screening OpenEnv."""
import sys
sys.path.insert(0, "/app")

from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, Request
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
    task: Optional[str] = None
    seed: Optional[int] = None


class StepRequest(BaseModel):
    episode_id: str
    action: Dict[str, Any]


@app.get("/health")
def health():
    return {"status": "ok", "service": "pe-deal-screening-env"}


@app.get("/info")
def info():
    return {
        "name": "pe_deal_screening_env",
        "version": "1.0.0",
        "tasks": env.TASKS,
        "description": "PE deal screening, IC memo writing, and portfolio prioritization.",
        "endpoints": ["/reset", "/step", "/health", "/info"],
    }


@app.post("/reset")
async def reset(request: Request):
    """Reset/start a new episode. Accepts optional JSON body with task and seed."""
    try:
        body = await request.body()
        task = None
        seed = None
        if body:
            import json
            try:
                data = json.loads(body)
                task = data.get("task", None)
                seed = data.get("seed", None)
            except Exception:
                pass
        result = env.reset(task=task, seed=seed)
        return result.model_dump()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")


@app.post("/step")
async def step(request: Request):
    """Submit an action and get reward."""
    try:
        body = await request.body()
        if not body:
            raise HTTPException(status_code=400, detail="Request body required")
        import json
        data = json.loads(body)
        episode_id = data.get("episode_id")
        action = data.get("action", {})
        if not episode_id:
            raise HTTPException(status_code=400, detail="episode_id required")
        result = env.step(episode_id=episode_id, action=action)
        return result.model_dump()
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")


@app.get("/episode/{episode_id}")
def get_episode(episode_id: str):
    ep = env.get_episode(episode_id)
    if ep is None:
        raise HTTPException(status_code=404, detail=f"Episode {episode_id} not found")
    return {
        "episode_id": episode_id,
        "task": ep["task"],
        "step": ep["step"],
        "done": ep["done"],
        "seed": ep["seed"],
    }
