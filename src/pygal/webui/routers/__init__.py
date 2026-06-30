"""pyGal WebUI 路由模块。"""

from .persona import router as persona_router
from .skills import router as skills_router
from .agent_trace import router as agent_trace_router
from .init_guide import router as init_guide_router

__all__ = [
    "persona_router",
    "skills_router",
    "agent_trace_router",
    "init_guide_router",
]
