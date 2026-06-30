"""ToolManager — 工具管理器。

负责:
  - 初始化 ToolRegistry 并注册内置工具
  - 对接 AgentNodes（提供工具查询/执行接口）
  - 管理工具启用/禁用状态
  - 记录工具调用历史（供 WebUI agent_trace 使用）
"""

from __future__ import annotations

from typing import Any, Optional

from .base import Tool, ToolRegistry, ToolResult
from .builtin import SearchWebTool, GetTimeTool, QueryLPMMTool


class ToolManager:
    """工具管理器 — Agent 与工具系统之间的桥梁。"""

    def __init__(self, lpmm_adapter=None, search_func=None):
        """初始化工具管理器。

        Args:
            lpmm_adapter: LPMM 适配器（可选，用于 QueryLPMMTool）
            search_func:  搜索函数（可选，用于 SearchWebTool 真实搜索）
        """
        self.registry = ToolRegistry()
        self._call_history: list[dict] = []

        # 注册内置工具
        self.registry.register(SearchWebTool(search_func=search_func), enabled=True)
        self.registry.register(GetTimeTool(), enabled=True)
        self.registry.register(QueryLPMMTool(lpmm_adapter=lpmm_adapter), enabled=True)

    @property
    def call_history(self) -> list[dict]:
        """工具调用历史记录。"""
        return self._call_history

    def get_enabled_tools(self) -> list[Tool]:
        """获取已启用的工具列表。"""
        return self.registry.list_enabled()

    def get_enabled_names(self) -> list[str]:
        """获取已启用的工具名称列表。"""
        return self.registry.list_enabled_names()

    def export_schemas(self) -> list[dict]:
        """导出已启用工具的 LLM schema。"""
        return self.registry.export_schemas()

    def execute(self, name: str, **kwargs) -> ToolResult:
        """执行工具并记录历史。"""
        result = self.registry.execute(name, **kwargs)

        self._call_history.append({
            "tool": name,
            "params": kwargs,
            "success": result.success,
            "output_preview": result.output[:200] if result.output else "",
            "error": result.error,
        })

        return result

    def set_enabled(self, name: str, enabled: bool) -> bool:
        """启用/禁用工具。"""
        if self.registry.get(name):
            self.registry.set_enabled(name, enabled)
            return True
        return False

    def list_all(self) -> list[dict]:
        """列出所有工具及状态。"""
        return self.registry.list_all()

    def clear_history(self) -> None:
        """清空调用历史。"""
        self._call_history.clear()
