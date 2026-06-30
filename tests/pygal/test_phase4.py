"""pyGal 阶段 4 测试 — Skill 调用系统。"""
from __future__ import annotations

import random
import pytest
from pathlib import Path

from pygal.tools.base import Tool, ToolRegistry, ToolResult
from pygal.tools.builtin import SearchWebTool, GetTimeTool, QueryLPMMTool
from pygal.tools.manager import ToolManager
from pygal.core.state import PyGalState, MessageEvent
from pygal.core.graph import PyGalAgent
from pygal.core.nodes import AgentNodes
from pygal.adapter.llm import MockLLMAdapter
from pygal.adapter.lpmm import MockLPMMAdapter
from pygal.adapter.platform import MockPlatformAdapter
from pygal.persona.model import PersonaConfig, BigFive, MBTI
from pygal.persona.renderer import PersonaRenderer


# ── 辅助 ──

def make_agent_with_tools() -> tuple[PyGalAgent, ToolManager]:
    """创建带 ToolManager 的 Agent。"""
    tm = ToolManager()
    renderer = PersonaRenderer()
    nodes = AgentNodes(
        llm=MockLLMAdapter(), lpmm=MockLPMMAdapter(),
        platform=MockPlatformAdapter(),
        persona_renderer=renderer, tool_manager=tm,
    )
    return PyGalAgent(nodes), tm


def make_state(msg_content="什么是 OCEAN？", extraversion=80, openness=85):
    config = PersonaConfig(
        big_five=BigFive(extraversion=extraversion, openness=openness, agreeableness=70),
        mbti=MBTI(ei="E"),
    )
    snapshot = PersonaRenderer().render(config)
    state = PyGalState()
    state.persona = snapshot
    state.messages = [MessageEvent(
        sender="u1", sender_name="Alice",
        content=msg_content, chat_id="g1", is_mention_me=True,
    )]
    return state


# ── 1. Tool 基类 ──

class TestToolBase:

    def test_tool_result_defaults(self):
        r = ToolResult(success=True)
        assert r.success is True
        assert r.output == ""
        assert r.data == {}
        assert r.error == ""

    def test_tool_result_to_dict(self):
        r = ToolResult(success=True, output="hello", data={"k": 1})
        d = r.to_dict()
        assert d["success"] is True
        assert d["output"] == "hello"
        assert d["data"]["k"] == 1


# ── 2. 内置工具 ──

class TestBuiltinTools:

    def test_search_web_mock(self):
        tool = SearchWebTool()
        result = tool.execute(query="Python tutorial")
        assert result.success
        assert "搜索结果" in result.output or "result" in result.output.lower()
        assert result.data["query"] == "Python tutorial"

    def test_search_web_requires_query(self):
        tool = SearchWebTool()
        result = tool.execute()
        assert not result.success

    def test_get_time(self):
        tool = GetTimeTool()
        result = tool.execute()
        assert result.success
        assert len(result.output) > 0
        assert "datetime" in result.data

    def test_get_time_with_timezone(self):
        tool = GetTimeTool()
        result = tool.execute(timezone="Asia/Shanghai")
        assert result.success

    def test_query_lpmm_mock(self):
        tool = QueryLPMMTool()
        result = tool.execute(query="用户偏好")
        assert result.success
        assert len(result.data["results"]) > 0

    def test_query_lpmm_with_adapter(self):
        lpmm = MockLPMMAdapter()
        tool = QueryLPMMTool(lpmm_adapter=lpmm)
        result = tool.execute(query="test")
        assert result.success

    def test_tool_schemas(self):
        tool = SearchWebTool()
        schema = tool.to_llm_schema()
        assert schema["name"] == "search_web"
        assert "description" in schema
        assert "parameters" in schema
        assert schema["parameters"]["type"] == "object"


# ── 3. ToolRegistry ──

class TestToolRegistry:

    def test_register_and_get(self):
        reg = ToolRegistry()
        reg.register(SearchWebTool(), enabled=True)
        tool = reg.get("search_web")
        assert tool is not None
        assert tool.name == "search_web"

    def test_list_enabled(self):
        reg = ToolRegistry()
        reg.register(SearchWebTool(), enabled=True)
        reg.register(GetTimeTool(), enabled=False)
        enabled = reg.list_enabled()
        assert len(enabled) == 1
        assert enabled[0].name == "search_web"

    def test_enable_disable(self):
        reg = ToolRegistry()
        reg.register(SearchWebTool(), enabled=True)
        assert reg.is_enabled("search_web")
        reg.set_enabled("search_web", False)
        assert not reg.is_enabled("search_web")

    def test_execute_disabled_tool(self):
        reg = ToolRegistry()
        reg.register(SearchWebTool(), enabled=False)
        result = reg.execute("search_web", query="test")
        assert not result.success
        assert "禁用" in result.error

    def test_execute_unknown_tool(self):
        reg = ToolRegistry()
        result = reg.execute("nonexistent", query="test")
        assert not result.success
        assert "不存在" in result.error

    def test_export_schemas(self):
        reg = ToolRegistry()
        reg.register(SearchWebTool(), enabled=True)
        reg.register(GetTimeTool(), enabled=True)
        schemas = reg.export_schemas()
        assert len(schemas) == 2
        names = [s["name"] for s in schemas]
        assert "search_web" in names
        assert "get_time" in names


# ── 4. ToolManager ──

class TestToolManager:

    def test_default_tools_registered(self):
        tm = ToolManager()
        names = [t["name"] for t in tm.list_all()]
        assert "search_web" in names
        assert "get_time" in names
        assert "query_lpmm" in names

    def test_all_enabled_by_default(self):
        tm = ToolManager()
        enabled = tm.get_enabled_names()
        assert len(enabled) == 3

    def test_disable_tool(self):
        tm = ToolManager()
        assert tm.set_enabled("search_web", False)
        assert "search_web" not in tm.get_enabled_names()

    def test_execute_records_history(self):
        tm = ToolManager()
        tm.execute("get_time")
        assert len(tm.call_history) == 1
        assert tm.call_history[0]["tool"] == "get_time"
        assert tm.call_history[0]["success"] is True

    def test_execute_search_records_params(self):
        tm = ToolManager()
        tm.execute("search_web", query="test query")
        assert tm.call_history[0]["params"]["query"] == "test query"

    def test_clear_history(self):
        tm = ToolManager()
        tm.execute("get_time")
        tm.clear_history()
        assert len(tm.call_history) == 0

    def test_export_schemas(self):
        tm = ToolManager()
        schemas = tm.export_schemas()
        assert len(schemas) == 3


# ── 5. Agent + ToolManager 集成 ──

class TestAgentToolIntegration:

    def test_agent_uses_tool_manager(self):
        """Agent 通过 ToolManager 执行搜索。"""
        random.seed(42)
        agent, tm = make_agent_with_tools()
        state = make_state("什么是大五人格模型？")
        result = agent.run(state)

        # 检查工具调用历史
        # search 可能有也可能没有（概率），但如果有应该记录在 ToolManager 中
        trace = result.get("trace", [])
        act_traces = [t for t in trace if t["node"] == "act"]
        for at in act_traces:
            if "tool_call" in at["data"]:
                assert at["data"]["tool_call"] == "search_web"

    def test_disabled_tool_not_used(self):
        """禁用的工具不会被使用。"""
        random.seed(42)
        agent, tm = make_agent_with_tools()
        tm.set_enabled("search_web", False)

        state = make_state("什么是 OCEAN？")
        result = agent.run(state)

        trace = result.get("trace", [])
        decide_traces = [t for t in trace if t["node"] == "decide"]
        # 不应该有 search 决策
        for dt in decide_traces:
            assert dt["data"]["action_type"] != "search"

    def test_tool_call_history_captured(self):
        """工具调用历史被正确捕获。"""
        random.seed(42)
        agent, tm = make_agent_with_tools()
        state = make_state("什么是 OCEAN？")
        agent.run(state)

        # 如果有搜索调用，历史应该记录
        for entry in tm.call_history:
            assert "tool" in entry
            assert "params" in entry
            assert "success" in entry

    def test_search_result_in_action(self):
        """搜索结果出现在 action_result 中。"""
        random.seed(42)
        agent, tm = make_agent_with_tools()
        state = make_state("什么是 OCEAN？")
        result = agent.run(state)

        # 检查 action_result 的 tool_calls
        ar = result.get("action_result")
        if ar and ar.tool_calls:
            for tc in ar.tool_calls:
                assert "tool" in tc
                assert "query" in tc
                assert "result_success" in tc
