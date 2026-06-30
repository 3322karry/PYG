"""内置技能 — 三个基础工具。

  - SearchWebTool:  搜索网络资讯
  - GetTimeTool:    获取当前时间
  - QueryLPMMTool:  查询 MaiBot LPMM 历史记忆

每个工具有 Mock 实现（独立测试）和真实接口骨架（后续对接 MaiBot）。
"""

from __future__ import annotations

import datetime
from typing import Any, Optional

from .base import Tool, ToolResult


# ── 1. SearchWeb ──────────────────────────────────────

class SearchWebTool(Tool):
    """网络搜索工具。

    Mock 模式: 返回预设结果
    真实模式: 对接 Brave Search / SerpAPI / MaiBot MCP 搜索
    """

    @property
    def name(self) -> str:
        return "search_web"

    @property
    def description(self) -> str:
        return (
            "搜索互联网获取最新信息。"
            "当用户问到你不了解的事实性问题、新闻、或需要实时数据时使用。"
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "搜索关键词",
                },
                "max_results": {
                    "type": "integer",
                    "description": "最大返回结果数",
                    "default": 3,
                },
            },
            "required": ["query"],
        }

    def __init__(self, search_func: Optional[callable] = None):
        """可选注入真实搜索函数。"""
        self._search_func = search_func

    def execute(self, query: str = "", max_results: int = 3, **kwargs) -> ToolResult:
        if not query:
            return ToolResult(success=False, error="缺少必选参数: query")

        if self._search_func:
            try:
                results = self._search_func(query, max_results)
                return ToolResult(
                    success=True,
                    output=self._format_results(query, results),
                    data={"query": query, "results": results},
                )
            except Exception as e:
                return ToolResult(success=False, error=str(e))

        # Mock 结果
        mock_results = [
            {"title": f"关于「{query}」的搜索结果 1", "snippet": f"这是与「{query}」相关的信息摘要……"},
            {"title": f"关于「{query}」的搜索结果 2", "snippet": f"另一条与「{query}」相关的信息……"},
            {"title": f"关于「{query}」的搜索结果 3", "snippet": f"更多关于「{query}」的补充信息……"},
        ][:max_results]

        return ToolResult(
            success=True,
            output=self._format_results(query, mock_results),
            data={"query": query, "results": mock_results},
        )

    @staticmethod
    def _format_results(query: str, results: list[dict]) -> str:
        lines = [f"搜索「{query}」的结果："]
        for i, r in enumerate(results, 1):
            lines.append(f"  {i}. {r.get('title', '')}")
            lines.append(f"     {r.get('snippet', '')}")
        return "\n".join(lines)


# ── 2. GetTime ────────────────────────────────────────

class GetTimeTool(Tool):
    """获取当前时间工具。"""

    @property
    def name(self) -> str:
        return "get_time"

    @property
    def description(self) -> str:
        return (
            "获取当前日期和时间。"
            "当用户询问时间、日期，或你需要判断时段（早/中/晚）时使用。"
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "timezone": {
                    "type": "string",
                    "description": "时区 (如 Asia/Shanghai)，默认系统时区",
                },
            },
            "required": [],
        }

    def execute(self, timezone: str = "", **kwargs) -> ToolResult:
        try:
            if timezone:
                tz = datetime.timezone(datetime.timedelta(hours=8))  # 简化
                now = datetime.datetime.now(tz)
            else:
                now = datetime.datetime.now()

            time_str = now.strftime("%Y年%m月%d日 %H:%M:%S")
            weekday = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"][now.weekday()]
            hour = now.hour

            if 6 <= hour < 12:
                period = "上午"
            elif 12 <= hour < 18:
                period = "下午"
            elif 18 <= hour < 24:
                period = "晚上"
            else:
                period = "深夜"

            output = f"现在是 {time_str} {weekday}（{period}）"

            return ToolResult(
                success=True,
                output=output,
                data={
                    "datetime": time_str,
                    "weekday": weekday,
                    "period": period,
                    "hour": hour,
                    "timestamp": now.timestamp(),
                },
            )
        except Exception as e:
            return ToolResult(success=False, error=str(e))


# ── 3. QueryLPMM ──────────────────────────────────────

class QueryLPMMTool(Tool):
    """查询 LPMM 历史记忆工具。

    封装 MaiBot A-Memorix 的语义检索能力。
    """

    @property
    def name(self) -> str:
        return "query_lpmm"

    @property
    def description(self) -> str:
        return (
            "查询历史记忆库，检索与当前话题相关的过往对话或用户信息。"
            "当你需要回忆之前聊过的内容、用户偏好、或历史事件时使用。"
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "查询文本（自然语言）",
                },
                "limit": {
                    "type": "integer",
                    "description": "返回条数",
                    "default": 5,
                },
            },
            "required": ["query"],
        }

    def __init__(self, lpmm_adapter=None):
        """可选注入 LPMM 适配器。"""
        self._lpmm = lpmm_adapter

    def execute(self, query: str, limit: int = 5, **kwargs) -> ToolResult:
        if self._lpmm:
            try:
                results = self._lpmm.search(query=query, limit=limit)
                output_lines = [f"找到 {len(results)} 条相关记忆："]
                for i, m in enumerate(results, 1):
                    content = m.get("content", "")
                    output_lines.append(f"  {i}. {content[:100]}")
                return ToolResult(
                    success=True,
                    output="\n".join(output_lines),
                    data={"query": query, "results": results},
                )
            except Exception as e:
                return ToolResult(success=False, error=str(e))

        # Mock 结果
        mock_memories = [
            {"content": f"之前聊过关于「{query}」的话题", "timestamp": "2025-06-01"},
            {"content": f"用户对「{query}」表现出兴趣", "timestamp": "2025-06-15"},
        ][:limit]

        output_lines = [f"找到 {len(mock_memories)} 条相关记忆："]
        for i, m in enumerate(mock_memories, 1):
            output_lines.append(f"  {i}. [{m['timestamp']}] {m['content']}")

        return ToolResult(
            success=True,
            output="\n".join(output_lines),
            data={"query": query, "results": mock_memories},
        )
