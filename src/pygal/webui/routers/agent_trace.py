"""Agent 追踪 API — GET /api/agent_trace。"""

from __future__ import annotations

from fastapi import APIRouter, Request

router = APIRouter()


@router.get("")
@router.get("/")
async def get_agent_trace(request: Request):
    """获取最近的 Agent 推理链追踪。"""
    state = request.app.state.pygal

    return {
        "trace": state.last_trace,
        "action_result": state.last_action_result,
        "internal_state": state.get_internal_state(),
    }


@router.get("/reflection")
async def get_reflection_history(request: Request):
    """获取反思历史。"""
    state = request.app.state.pygal

    if state.reflection_engine:
        return {"history": state.reflection_engine.get_history_summary()}
    return {"history": []}


@router.get("/state")
async def get_agent_internal_state(request: Request):
    """获取 Agent 内部状态（精力/无聊度等）。"""
    state = request.app.state.pygal

    return {
        "internal": state.get_internal_state(),
        "initialized": state.initialized,
    }
