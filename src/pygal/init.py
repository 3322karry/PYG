"""pyGal — MaiBot 集成入口。

在 MaiBot 启动时调用 pygal.init() 初始化 pyGal Agent 系统。
pyGal 会替换 Maisaka 推理引擎，接管对话循环。

使用方式:
    from pygal import init as init_pygal
    await init_pygal()
"""

from __future__ import annotations

import logging
from typing import Optional

from .core.graph import PyGalAgent
from .core.nodes import AgentNodes
from .adapter.llm import MaiBotLLMAdapter, MockLLMAdapter
from .adapter.lpmm import MaiBotLPMMAdapter, MockLPMMAdapter
from .adapter.platform import MaiBotPlatformAdapter, MockPlatformAdapter
from .persona.model import PersonaConfig
from .persona.renderer import PersonaRenderer
from .tools.manager import ToolManager
from .reflection.engine import ReflectionEngine
from .scheduler import ActionScheduler, SchedulerConfig
from .webui.state import WebUIState

logger = logging.getLogger("pygal")

__all__ = ["init", "get_agent", "get_tool_manager", "get_reflection_engine"]

# 全局实例
_agent: Optional[PyGalAgent] = None
_tool_manager: Optional[ToolManager] = None
_reflection_engine: Optional[ReflectionEngine] = None
_scheduler: Optional[ActionScheduler] = None


async def init(
    persona_config_path: str = "config/pygal/persona.json",
    use_mock: bool = False,
) -> WebUIState:
    """初始化 pyGal Agent 系统。

    Args:
        persona_config_path: 人格配置文件路径
        use_mock: 是否使用 Mock 适配器（独立测试模式）

    Returns:
        WebUIState 实例，供 WebUI 集成使用
    """
    global _agent, _tool_manager, _reflection_engine, _scheduler

    logger.info("pyGal 初始化中...")

    # 1. 加载人格配置
    from pathlib import Path
    config_path = Path(persona_config_path)
    if config_path.exists():
        persona_config = PersonaConfig.from_file(config_path)
    else:
        logger.warning(f"人格配置不存在: {config_path}，使用默认配置")
        persona_config = PersonaConfig()

    renderer = PersonaRenderer()
    snapshot = renderer.render(persona_config)
    logger.info(f"人格已加载: {persona_config.name} ({persona_config.nickname})")

    # 2. 初始化适配器
    if use_mock:
        llm = MockLLMAdapter()
        lpmm = MockLPMMAdapter()
        platform = MockPlatformAdapter()
    else:
        llm = MaiBotLLMAdapter(task_name="chat")
        lpmm = MaiBotLPMMAdapter()
        platform = MaiBotPlatformAdapter()

    # 3. 初始化工具管理器
    _tool_manager = ToolManager(lpmm_adapter=lpmm)

    # 4. 初始化反思引擎
    _reflection_engine = ReflectionEngine(llm=llm, lpmm=lpmm)

    # 5. 初始化调度器
    _scheduler = ActionScheduler(SchedulerConfig())
    _scheduler.set_persona(snapshot)

    # 6. 构建 Agent
    nodes = AgentNodes(
        llm=llm,
        lpmm=lpmm,
        platform=platform,
        persona_renderer=renderer,
        tool_manager=_tool_manager,
        reflection_engine=_reflection_engine,
    )
    _agent = PyGalAgent(nodes)

    logger.info("pyGal 初始化完成！")

    # 7. 构建 WebUI 状态
    state = WebUIState(
        agent=_agent,
        tool_manager=_tool_manager,
        reflection_engine=_reflection_engine,
        scheduler=_scheduler,
        persona_config_path=config_path,
        initialized=True,
    )

    return state


def get_agent() -> Optional[PyGalAgent]:
    """获取全局 Agent 实例。"""
    return _agent


def get_tool_manager() -> Optional[ToolManager]:
    """获取全局 ToolManager 实例。"""
    return _tool_manager


def get_reflection_engine() -> Optional[ReflectionEngine]:
    """获取全局 ReflectionEngine 实例。"""
    return _reflection_engine
