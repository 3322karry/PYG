"""技能管理 API — GET/PUT /api/skills。"""

from __future__ import annotations

from fastapi import APIRouter, Request
from pydantic import BaseModel

router = APIRouter()


class SkillToggle(BaseModel):
    """技能开关请求体。"""
    name: str
    enabled: bool


@router.get("")
@router.get("/")
async def list_skills(request: Request):
    """列出所有可用技能及状态。"""
    state = request.app.state.pygal

    if state.tool_manager:
        return {"skills": state.tool_manager.list_all()}
    return {"skills": []}


@router.put("/{skill_name}")
async def toggle_skill(skill_name: str, payload: SkillToggle, request: Request):
    """启用/禁用技能。"""
    state = request.app.state.pygal

    if not state.tool_manager:
        return {"status": "error", "message": "ToolManager 未初始化"}

    success = state.tool_manager.set_enabled(skill_name, payload.enabled)
    if not success:
        return {"status": "error", "message": f"技能不存在: {skill_name}"}

    return {"status": "ok", "skill": skill_name, "enabled": payload.enabled}


@router.get("/history")
async def skill_call_history(request: Request):
    """获取工具调用历史。"""
    state = request.app.state.pygal

    if state.tool_manager:
        return {"history": state.tool_manager.call_history}
    return {"history": []}
