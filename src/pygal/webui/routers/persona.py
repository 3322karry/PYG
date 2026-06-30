"""人格配置 API — GET/POST /api/persona。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel, Field

from pygal.persona.model import PersonaConfig, BigFive, MBTI
from pygal.persona.renderer import PersonaRenderer

if TYPE_CHECKING:
    from ..state import WebUIState

router = APIRouter()


class PersonaPayload(BaseModel):
    """人格配置请求体。"""
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
    speech_style_override: str | None = None


class PersonaPreview(BaseModel):
    """人格渲染预览。"""
    system_prompt: str
    action_tendency: float
    reply_willingness: float
    topic_initiative: float
    lurk_tendency: float
    emoji_frequency: float
    emotional_volatility: float
    curiosity_drive: float
    formality: float
    tone_style: str
    speech_style_hint: str


def _render_preview(snapshot) -> PersonaPreview:
    """从 PersonaSnapshot 创建预览。"""
    return PersonaPreview(
        system_prompt=snapshot.system_prompt,
        action_tendency=snapshot.action_tendency,
        reply_willingness=snapshot.reply_willingness,
        topic_initiative=snapshot.topic_initiative,
        lurk_tendency=snapshot.lurk_tendency,
        emoji_frequency=snapshot.emoji_frequency,
        emotional_volatility=snapshot.emotional_volatility,
        curiosity_drive=snapshot.curiosity_drive,
        formality=snapshot.formality,
        tone_style=snapshot.tone_style,
        speech_style_hint=snapshot.speech_style_hint,
    )


@router.get("")
@router.get("/")
async def get_persona(request: Request):
    """读取当前人格配置。"""
    state = request.app.state.pygal
    path = state.persona_config_path

    if not path.exists():
        return {"persona": PersonaPayload().model_dump(), "preview": None}

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    config = PersonaConfig.from_dict(data)
    renderer = PersonaRenderer()
    snapshot = renderer.render(config)

    return {"persona": data, "preview": _render_preview(snapshot).model_dump()}


@router.post("")
@router.post("/")
async def save_persona(payload: PersonaPayload, request: Request):
    """保存人格配置。"""
    state = request.app.state.pygal
    path = state.persona_config_path

    path.parent.mkdir(parents=True, exist_ok=True)
    config = PersonaConfig.from_dict(payload.model_dump())
    config.to_file(path)

    renderer = PersonaRenderer()
    snapshot = renderer.render(config)

    return {"status": "ok", "preview": _render_preview(snapshot).model_dump()}


@router.post("/preview")
async def preview_persona(payload: PersonaPayload, request: Request):
    """实时预览人格渲染结果（不保存）。"""
    config = PersonaConfig.from_dict(payload.model_dump())
    renderer = PersonaRenderer()
    snapshot = renderer.render(config)

    return _render_preview(snapshot).model_dump()
