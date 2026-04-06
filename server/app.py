"""FastAPI application for PE Deal Screening OpenEnv."""
import os
import sys
sys.path.insert(0, "/app")

from typing import Any, Dict, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from pydantic import BaseModel

import server.environment as env
from pe_env.models import (
    DealScreeningObservation,
    ICMemoObservation,
    PortfolioPrioritizationObservation,
    StepResult,
    ResetResponse,
)

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


@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/docs")


class ResetRequest(BaseModel):
    task: Optional[str] = None
    seed: Optional[int] = None


class StepRequest(BaseModel):
    episode_id: str
    action: Dict[str, Any]


@app.get("/health")
def health():
    """Health check endpoint - must return 'healthy'."""
    return {"status": "healthy", "service": "pe-deal-screening-env"}


@app.get("/info")
def info():
    """Environment info."""
    return {
        "name": "pe_deal_screening_env",
        "version": "1.0.0",
        "tasks": env.TASKS,
        "description": "PE deal screening, IC memo writing, and portfolio prioritization.",
        "endpoints": ["/reset", "/step", "/health", "/info", "/metadata", "/schema", "/mcp"],
    }


@app.get("/metadata")
def metadata():
    """Environment metadata."""
    return {
        "name": "pe_deal_screening_env",
        "description": "OpenEnv environment for PE deal screening, IC memo writing & portfolio prioritization.",
        "version": "1.0.0",
        "tasks": env.TASKS,
        "author": "jAIdeep-Murthy",
        "tags": ["finance", "private-equity", "decision-making", "multi-task"],
    }


@app.get("/schema")
def schema():
    """Return unified action, observation, and state schemas for OpenEnv compliance."""
    deal_obs = DealScreeningObservation.model_json_schema()
    deal_action = {
        "type": "object",
        "properties": {
            "decision": {"type": "string", "enum": ["INVEST", "PASS"]},
            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "rationale": {"type": "string"},
        },
        "required": ["decision", "confidence", "rationale"],
    }
    return {
        "action": deal_action,
        "observation": deal_obs,
        "state": {
            "type": "object",
            "description": "Current environment state",
            "properties": {
                "episode_id": {"type": "string"},
                "task": {"type": "string"},
                "step": {"type": "integer"},
                "done": {"type": "boolean"},
                "observation": {"type": "object"},
            },
        },
        "tasks": {
            "deal_screening": {
                "observation": deal_obs,
                "action": deal_action,
            },
            "ic_memo": {
                "observation": ICMemoObservation.model_json_schema(),
                "action": {
                    "type": "object",
                    "properties": {
                        "executive_summary": {"type": "string"},
                        "investment_thesis": {"type": "string"},
                        "key_risks": {"type": "string"},
                        "financial_highlights": {"type": "string"},
                        "recommendation": {"type": "string", "enum": ["INVEST", "PASS", "CONDITIONAL"]},
                    },
                    "required": ["executive_summary", "investment_thesis", "key_risks", "financial_highlights", "recommendation"],
                },
            },
            "portfolio_prioritization": {
                "observation": PortfolioPrioritizationObservation.model_json_schema(),
                "action": {
                    "type": "object",
                    "properties": {
                        "allocations": {"type": "array"},
                        "total_deployed_mm": {"type": "number"},
                        "rationale": {"type": "string"},
                    },
                    "required": ["allocations", "total_deployed_mm", "rationale"],
                },
            },
        },
    }


@app.get("/state")
def get_state():
    """Get current environment state."""
    return env.get_state()


@app.post("/mcp")
async def mcp(request: Request):
    """MCP endpoint for tool discovery - returns JSON-RPC 2.0 payload."""
    try:
        payload = await request.json()
    except Exception:
        payload = {}
    return {
        "jsonrpc": "2.0",
        "id": payload.get("id") if isinstance(payload, dict) else None,
        "result": {
            "name": "pe_deal_screening_env",
            "tools": ["reset", "step", "state"],
        },
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
    """Get episode state."""
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


def main():
    """Entry point for server script (required by pyproject.toml [project.scripts])."""
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("server.app:app", host="0.0.0.0", port=port, reload=False)


if __name__ == "__main__":
    main()
