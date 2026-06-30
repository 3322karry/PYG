"""pyGal 技能系统 — 基于 LangChain Tool Calling 标准的工具封装。

将外部能力封装为标准 Tool 对象，Agent 在推理过程中自主决定是否调用。
"""

from .base import Tool, ToolRegistry, ToolResult
from .builtin import SearchWebTool, GetTimeTool, QueryLPMMTool
from .manager import ToolManager

__all__ = [
    "Tool", "ToolRegistry", "ToolResult",
    "SearchWebTool", "GetTimeTool", "QueryLPMMTool",
    "ToolManager",
]
