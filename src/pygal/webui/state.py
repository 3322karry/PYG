"""WebUI 共享状态 — 持有 Agent 各组件的引用。

所有路由通过 `request.app.state.pygal` 访问共享状态。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Any
from pathlib import Path


@dataclass
class WebUIState:
    """WebUI 共享状态。

    持有 pyGal 各组件的引用，使路由可以读取/修改运行时状态。
    """

    # 核心组件
    agent: Any = None              # PyGalAgent 实例
    tool_manager: Any = None       # ToolManager 实例
    reflection_engine: Any = None  # ReflectionEngine 实例
    scheduler: Any = None          # ActionScheduler 实例

    # 配置
    persona_config_path: Path = Path("config/persona.json")
    initialized: bool = False

    # 最近一次 Agent 运行的追踪
    last_trace: list[dict] = field(default_factory=list)
    last_action_result: Optional[dict] = None
    last_internal_state: Optional[dict] = None

    def update_after_agent_run(self, result: dict) -> None:
        """Agent 运行后更新追踪状态。"""
        self.last_trace = result.get("trace", [])
        ar = result.get("action_result")
        if ar is None:
            self.last_action_result = None
        elif hasattr(ar, "to_dict"):
            self.last_action_result = ar.to_dict()
        elif hasattr(ar, "__dict__"):
            self.last_action_result = {
                "action_type": getattr(ar, "action_type", ""),
                "content": getattr(ar, "content", ""),
                "reasoning": getattr(ar, "reasoning", ""),
                "tool_calls": getattr(ar, "tool_calls", []),
            }
        elif isinstance(ar, dict):
            self.last_action_result = ar
        else:
            self.last_action_result = None

        internal = result.get("internal")
        if internal and hasattr(internal, "to_dict"):
            self.last_internal_state = internal.to_dict()

    def get_internal_state(self) -> dict:
        """获取当前内部状态。"""
        if self.scheduler:
            return self.scheduler._internal.to_dict()
        return self.last_internal_state or {}
