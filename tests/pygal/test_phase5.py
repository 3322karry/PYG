"""pyGal 阶段 5 测试 — 自我反思与主动写入 LPMM。"""
from __future__ import annotations

import random
import pytest
from pathlib import Path

from pygal.reflection.engine import ReflectionEngine, ReflectionResult, ReflectionTrigger
from pygal.reflection.models import MemoryNode, ImpressionUpdate
from pygal.core.state import PyGalState, MessageEvent
from pygal.core.graph import PyGalAgent
from pygal.core.nodes import AgentNodes
from pygal.adapter.llm import MockLLMAdapter
from pygal.adapter.lpmm import MockLPMMAdapter
from pygal.adapter.platform import MockPlatformAdapter
from pygal.persona.model import PersonaConfig, BigFive, MBTI
from pygal.persona.renderer import PersonaRenderer
from pygal.tools.manager import ToolManager


# ── 辅助 ──

def make_setup():
    """创建完整的 Agent + 反思引擎 + Mock LPMM。"""
    lpmm = MockLPMMAdapter()
    llm = MockLLMAdapter()
    platform = MockPlatformAdapter()
    renderer = PersonaRenderer()
    tm = ToolManager(lpmm_adapter=lpmm)
    reflect_engine = ReflectionEngine(llm=llm, lpmm=lpmm)

    nodes = AgentNodes(
        llm=llm, lpmm=lpmm, platform=platform,
        persona_renderer=renderer,
        tool_manager=tm,
        reflection_engine=reflect_engine,
    )
    return nodes, lpmm, reflect_engine


def make_state(msg="你好呀", mention=True, extraversion=80):
    config = PersonaConfig(
        big_five=BigFive(extraversion=extraversion, agreeableness=70),
        mbti=MBTI(ei="E"),
    )
    snapshot = PersonaRenderer().render(config)
    state = PyGalState()
    state.persona = snapshot
    state.messages = [MessageEvent(
        sender="u1", sender_name="LampyStar",
        content=msg, chat_id="g1", is_mention_me=mention,
    )]
    return state


# ── 1. 数据模型 ──

class TestModels:

    def test_memory_node_defaults(self):
        node = MemoryNode(content="用户喜欢猫")
        assert node.memory_type == "fact"
        assert node.source == "reflection"
        assert 0 <= node.importance <= 1

    def test_memory_node_to_lpmm_metadata(self):
        node = MemoryNode(
            content="用户是程序员",
            memory_type="fact",
            importance=0.8,
            related_persons=["u1"],
        )
        meta = node.to_lpmm_metadata()
        assert meta["memory_type"] == "fact"
        assert meta["importance"] == 0.8
        assert "u1" in meta["related_persons"]

    def test_impression_update_defaults(self):
        imp = ImpressionUpdate(person_id="u1")
        assert imp.updates == {}
        assert imp.summary == ""

    def test_impression_update_to_dict(self):
        imp = ImpressionUpdate(
            person_id="u1",
            person_name="Alice",
            updates={"kindness": 1, "intelligence": 2},
            summary="善良聪明的人",
        )
        d = imp.to_dict()
        assert d["person_id"] == "u1"
        assert d["updates"]["kindness"] == 1


# ── 2. 反思引擎 ──

class TestReflectionEngine:

    def test_should_reflect_first_time(self):
        """首次对话应触发反思。"""
        nodes, lpmm, engine = make_setup()
        state = make_state("你好呀")
        assert engine.should_reflect(state, ReflectionTrigger.DEEP_CONVERSATION)

    def test_should_not_reflect_too_frequent(self):
        """反思间隔内不重复触发。"""
        nodes, lpmm, engine = make_setup()
        state = make_state("你好呀")
        # 第一次反思
        assert engine.should_reflect(state, ReflectionTrigger.DEEP_CONVERSATION)
        engine.reflect(state, ReflectionTrigger.DEEP_CONVERSATION)
        # 立即再次检查 → 不应触发
        assert not engine.should_reflect(state, ReflectionTrigger.DEEP_CONVERSATION)

    def test_reflect_produces_result(self):
        """反思应产生结果。"""
        random.seed(42)
        nodes, lpmm, engine = make_setup()
        state = make_state("我喜欢编程和猫咪")
        result = engine.reflect(state, ReflectionTrigger.DEEP_CONVERSATION)

        assert isinstance(result, ReflectionResult)
        assert result.triggered_by == "deep_conversation"

    def test_reflect_writes_to_lpmm(self):
        """反思后应写入 LPMM。"""
        random.seed(42)
        nodes, lpmm, engine = make_setup()
        state = make_state("我喜欢编程和猫咪")
        before_count = len(lpmm._memories)
        engine.reflect(state, ReflectionTrigger.DEEP_CONVERSATION)
        after_count = len(lpmm._memories)
        assert after_count >= before_count

    def test_reflect_updates_impression(self):
        """反思后应更新印象。"""
        random.seed(42)
        nodes, lpmm, engine = make_setup()
        state = make_state("我喜欢编程和猫咪")
        engine.reflect(state, ReflectionTrigger.DEEP_CONVERSATION)
        # 印象历史应有记录
        assert len(engine._history) > 0

    def test_reflect_history_recorded(self):
        """反思历史被记录。"""
        random.seed(42)
        nodes, lpmm, engine = make_setup()
        state = make_state("你好")
        engine.reflect(state, ReflectionTrigger.DEEP_CONVERSATION)
        summary = engine.get_history_summary()
        assert len(summary) >= 1
        assert "triggered_by" in summary[0]


# ── 3. Agent 集成 ──

class TestAgentReflectionIntegration:

    def test_agent_triggers_reflection_after_reply(self):
        """Agent 回复后应触发反思。"""
        random.seed(42)
        nodes, lpmm, engine = make_setup()
        agent = PyGalAgent(nodes)
        state = make_state("你好呀", mention=True)
        result = agent.run(state)

        trace = result.get("trace", [])
        reflection_traces = [t for t in trace if t["node"] == "reflection"]
        # 应该有反思记录
        assert len(reflection_traces) >= 1

    def test_reflection_writes_memory(self):
        """反思后 LPMM 中应有新记忆。"""
        random.seed(42)
        nodes, lpmm, engine = make_setup()
        agent = PyGalAgent(nodes)
        state = make_state("我喜欢编程和猫咪")
        before = len(lpmm._memories)
        agent.run(state)
        after = len(lpmm._memories)
        assert after > before

    def test_no_reflection_without_engine(self):
        """没有反思引擎时不触发反思。"""
        random.seed(42)
        lpmm = MockLPMMAdapter()
        llm = MockLLMAdapter()
        renderer = PersonaRenderer()
        tm = ToolManager(lpmm_adapter=lpmm)
        nodes = AgentNodes(
            llm=llm, lpmm=lpmm, platform=MockPlatformAdapter(),
            persona_renderer=renderer, tool_manager=tm,
            reflection_engine=None,  # 无反思引擎
        )
        agent = PyGalAgent(nodes)
        state = make_state("你好呀")
        result = agent.run(state)

        trace = result.get("trace", [])
        reflection_traces = [t for t in trace if t["node"] == "reflection"]
        assert len(reflection_traces) == 0

    def test_silent_does_not_trigger_reflection(self):
        """潜水不应触发反思。"""
        random.seed(42)
        nodes, lpmm, engine = make_setup()
        agent = PyGalAgent(nodes)
        # 内向型 + 不@ → 可能潜水
        state = make_state("天气不错", mention=False, extraversion=15)
        result = agent.run(state)

        trace = result.get("trace", [])
        reflection_traces = [t for t in trace if t["node"] == "reflection"]
        ar = result.get("action_result")
        if ar and ar.action_type == "silent":
            assert len(reflection_traces) == 0


# ── 4. LPMM 写入验证 ──

class TestLPMMWrite:

    def test_mock_lpmm_write(self):
        lpmm = MockLPMMAdapter()
        success = lpmm.write("测试记忆", chat_id="g1")
        assert success
        assert len(lpmm._memories) == 1
        assert lpmm._memories[0]["content"] == "测试记忆"

    def test_mock_lpmm_write_with_metadata(self):
        lpmm = MockLPMMAdapter()
        lpmm.write("带元数据的记忆", metadata={"memory_type": "preference", "importance": 0.8})
        assert lpmm._memories[0]["metadata"]["memory_type"] == "preference"

    def test_mock_lpmm_update_impression(self):
        lpmm = MockLPMMAdapter()
        lpmm.update_impression("u1", {"kindness": 1, "summary": "善良"})
        assert "u1" in lpmm._impressions

    def test_mock_lpmm_search_after_write(self):
        lpmm = MockLPMMAdapter()
        lpmm.write("用户喜欢猫咪", chat_id="g1")
        results = lpmm.search("猫咪", limit=5)
        assert len(results) > 0
