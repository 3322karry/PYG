"""pyGal WebUI — FastAPI 后端 API + 前端页面。

阶段 6 实现:
  后端 API:
    - /api/persona       (GET/POST)  人格配置读写
    - /api/skills        (GET/PUT)   技能管理
    - /api/agent_trace   (GET)       Agent 推理链追踪
    - /api/init_guide    (POST)      初始化向导
    - /api/agent_state   (GET)       内部状态（精力/无聊度等）
    - /api/reflection    (GET)       反思历史

  前端页面:
    - 配置页（大五人格滑块 + MBTI + 实时预览）
    - 主动行为与状态面板
    - 技能管理页
    - Agent 追踪与记忆页
    - 初始化向导
"""

from .app import create_app

__all__ = ["create_app"]
