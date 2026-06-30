"""初始化向导 API — POST /api/init_guide。"""

from __future__ import annotations

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from pygal.persona.model import PersonaConfig, BigFive, MBTI
from pygal.persona.renderer import PersonaRenderer
from pygal.tools.manager import ToolManager

router = APIRouter()


class InitGuidePayload(BaseModel):
    """初始化向导提交数据。"""
    # API Key 配置
    api_key: str = ""
    api_base: str = ""
    model_name: str = "deepseek-chat"

    # 人格配置
    name: str = "Galatea"
    nickname: str = "伽拉"
    big_five: dict = Field(
        default_factory=lambda: {
            "openness": 50, "conscientiousness": 50,
            "extraversion": 50, "agreeableness": 50, "neuroticism": 50,
        }
    )
    mbti: dict = Field(
        default_factory=lambda: {"ei": "I", "sn": "N", "tf": "F", "jp": "P"}
    )
    background: str = ""
    interests: list[str] = Field(default_factory=list)

    # 技能选择
    enabled_skills: list[str] = Field(
        default_factory=lambda: ["search_web", "get_time", "query_lpmm"]
    )


@router.post("")
@router.post("/")
async def submit_init_guide(payload: InitGuidePayload, request: Request):
    """提交初始化向导配置。"""
    state = request.app.state.pygal

    # 1. 保存人格配置
    config = PersonaConfig(
        name=payload.name,
        nickname=payload.nickname,
        big_five=BigFive.from_dict(payload.big_five),
        mbti=MBTI.from_dict(payload.mbti),
        background=payload.background,
        interests=payload.interests,
    )
    config.to_file(state.persona_config_path)

    # 2. 配置技能开关
    if state.tool_manager:
        all_skills = [s["name"] for s in state.tool_manager.list_all()]
        for skill_name in all_skills:
            should_enable = skill_name in payload.enabled_skills
            state.tool_manager.set_enabled(skill_name, should_enable)

    # 3. 标记已初始化
    state.initialized = True

    # 4. 渲染预览
    renderer = PersonaRenderer()
    snapshot = renderer.render(config)

    return {
        "status": "ok",
        "message": f"初始化完成！{payload.name}（{payload.nickname}）已就绪。",
        "persona_preview": {
            "system_prompt": snapshot.system_prompt[:200] + "...",
            "tone_style": snapshot.tone_style,
            "action_tendency": snapshot.action_tendency,
        },
        "enabled_skills": payload.enabled_skills,
        "api_configured": bool(payload.api_key),
    }


@router.get("/status")
async def init_status(request: Request):
    """检查是否需要初始化。"""
    state = request.app.state.pygal
    return {"initialized": state.initialized}
