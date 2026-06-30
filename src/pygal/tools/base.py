"""Tool 基类与注册表 — 标准化的工具抽象。

每个 Tool 包含:
  - name:        工具名称（唯一标识）
  - description: 工具描述（供 LLM 理解何时使用）
  - parameters:  参数 schema（JSON Schema 格式）
  - execute():   执行函数

ToolRegistry 管理所有已注册的工具，支持:
  - register / unregister
  - get / list
  - 按启用状态过滤
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Optional


@dataclass
class ToolResult:
    """工具执行结果。"""
    success: bool
    output: str = ""           # 工具输出（供 LLM 阅读）
    data: dict = field(default_factory=dict)  # 原始数据（供程序逻辑使用）
    error: str = ""

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "output": self.output,
            "data": self.data,
            "error": self.error,
        }


class Tool(ABC):
    """工具抽象基类。

    子类需实现:
      - name: 工具唯一标识
      - description: 供 LLM 理解工具用途
      - parameters: JSON Schema 参数定义
      - execute(): 实际执行逻辑
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """工具名称。"""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """工具描述（供 LLM 决定是否调用）。"""
        ...

    @property
    @abstractmethod
    def parameters(self) -> dict:
        """参数 schema (JSON Schema 格式)。"""
        ...

    @abstractmethod
    def execute(self, **kwargs) -> ToolResult:
        """执行工具。

        Args:
            **kwargs: 与 parameters schema 匹配的参数

        Returns:
            ToolResult
        """
        ...

    def to_llm_schema(self) -> dict:
        """转为 LLM function calling 格式。"""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }


class ToolRegistry:
    """工具注册表 — 管理所有可用工具。

    特性:
      - 工具注册/注销
      - 按名称查询
      - 按启用状态过滤
      - 导出 LLM 可用的 schema 列表
    """

    def __init__(self):
        self._tools: dict[str, Tool] = {}
        self._enabled: dict[str, bool] = {}

    def register(self, tool: Tool, enabled: bool = True) -> None:
        """注册一个工具。"""
        self._tools[tool.name] = tool
        self._enabled[tool.name] = enabled

    def unregister(self, name: str) -> bool:
        """注销工具。"""
        if name in self._tools:
            del self._tools[name]
            del self._enabled[name]
            return True
        return False

    def get(self, name: str) -> Tool | None:
        """获取工具。"""
        return self._tools.get(name)

    def is_enabled(self, name: str) -> bool:
        """检查工具是否启用。"""
        return self._enabled.get(name, False)

    def set_enabled(self, name: str, enabled: bool) -> None:
        """启用/禁用工具。"""
        if name in self._enabled:
            self._enabled[name] = enabled

    def list_all(self) -> list[dict]:
        """列出所有工具（含启用状态）。"""
        return [
            {
                "name": t.name,
                "description": t.description,
                "enabled": self._enabled.get(t.name, False),
            }
            for t in self._tools.values()
        ]

    def list_enabled(self) -> list[Tool]:
        """列出所有已启用的工具。"""
        return [
            t for name, t in self._tools.items()
            if self._enabled.get(name, False)
        ]

    def list_enabled_names(self) -> list[str]:
        """列出已启用工具的名称。"""
        return [t.name for t in self.list_enabled()]

    def export_schemas(self) -> list[dict]:
        """导出已启用工具的 LLM schema 列表。"""
        return [t.to_llm_schema() for t in self.list_enabled()]

    def execute(self, name: str, **kwargs) -> ToolResult:
        """执行指定工具。

        Args:
            name: 工具名称
            **kwargs: 工具参数

        Returns:
            ToolResult
        """
        tool = self.get(name)
        if not tool:
            return ToolResult(success=False, error=f"工具不存在: {name}")
        if not self.is_enabled(name):
            return ToolResult(success=False, error=f"工具已禁用: {name}")
        try:
            return tool.execute(**kwargs)
        except Exception as e:
            return ToolResult(success=False, error=str(e))
