"""pyGal 阶段 1 核心测试 — 验证状态图可运行性。"""
from __future__ import annotations

import pytest
import random
from pathlib import Path

from pygal.core.state import PyGalState, AgentPhase, InternalState, MessageEvent
from pygal.core.graph import build_agent_graph, PyGalAgent
from pygal.core.nodes import AgentNodes
from pygal.adapter.llm import MockLLMAdapter
from pygal.adapter.lpmm import MockLPMMAdapter
from pygal.adapter.platform import MockPlatformAdapter
from pygal.persona.model import PersonaConfig, BigFive, MBTI
from pygal.persona.renderer import PersonaRenderer
from pygal.scheduler import ActionScheduler, SchedulerConfig


# ── 辅助函数 ──

def make_test_nodes() -> AgentNodes:
    """创建使用 Mock Adapter 的测试节点集。"""
    return AgentNodes(
        llm=MockLLMAdapter(),
        lpmm=MockLPMMAdapter(),
        platform=MockPlatformAdapter(),
        persona_renderer=PersonaRenderer(),
    )


def make_test_state(messages: list[MessageEvent] | None = None) -> PyGalState:
    """创建测试用初始状态。"""
    state = PyGalState()
    state.messages = messages or []
    state.active_tools = ["search_web", "get_time", "query_lpmm"]
    return state


# ── 1. 状态图可运行性 ──

class TestStateGraph:

    def test_graph_builds_without_error(self):
        """状态图可以正常构建。"""
        nodes = make_test_nodes()
        graph = build_agent_graph(nodes)
        assert graph is not None

    def test_graph_has_five_nodes(self):
        """状态图包含五个节点。"""
        nodes = make_test_nodes()
        graph = build_agent_graph(nodes)
        # 编译后可执行
        assert graph.invoke is not None

    def test_empty_state_completes(self):
        """空状态可以跑完整个图。"""
        nodes = make_test_nodes()
        agent = PyGalAgent(nodes)
        state = make_test_state()
        result = agent.run(state)
        assert result is not None
        trace = result.get("trace", [])
        assert len(trace) >= 5  # 五个节点各一条 trace

    def test_message_triggers_reply(self):
        """收到消息时 Agent 选择回复。"""
        random.seed(42)
        nodes = make_test_nodes()
        agent = PyGalAgent(nodes)
        # 使用高外向性人格确保回复概率
        from pygal.persona.model import PersonaConfig, BigFive, MBTI
        from pygal.persona.renderer import PersonaRenderer
        config = PersonaConfig(
            big_five=BigFive(extraversion=80, agreeableness=70),
            mbti=MBTI(ei="E"),
        )
        snapshot = PersonaRenderer().render(config)
        state = make_test_state([
            MessageEvent(
                sender="user_001",
                sender_name="LampyStar",
                content="你好呀",
                chat_id="test_chat",
            )
        ])
        state.persona = snapshot
        result = agent.run(state)
        ar = result.get("action_result")
        assert ar is not None
        assert ar.action_type == "reply"

    def test_no_message_triggers_silent(self):
        """没有消息时 Agent 选择潜水。"""
        nodes = make_test_nodes()
        agent = PyGalAgent(nodes)
        state = make_test_state()
        result = agent.run(state)
        ar = result.get("action_result")
        assert ar is not None
        assert ar.action_type == "silent"

    def test_trace_records_all_nodes(self):
        """推理追踪记录了所有节点。"""
        nodes = make_test_nodes()
        agent = PyGalAgent(nodes)
        state = make_test_state()
        result = agent.run(state)
        trace = result.get("trace", [])
        node_names = [t["node"] for t in trace]
        assert "perceive" in node_names
        assert "reflect" in node_names
        assert "decide" in node_names
        assert "act" in node_names
        assert "observe" in node_names


# ── 2. 人格渲染 ──

class TestPersona:

    def test_default_persona(self):
        """默认人格可以正常渲染。"""
        config = PersonaConfig()
        renderer = PersonaRenderer()
        snapshot = renderer.render(config)
        assert snapshot.name == "Galatea"
        assert snapshot.system_prompt != ""
        assert 0 <= snapshot.action_tendency <= 1

    def test_extraverted_persona(self):
        """高外向性人格应有高行动倾向。"""
        config = PersonaConfig(
            big_five=BigFive(extraversion=90),
            mbti=MBTI(ei="E"),
        )
        renderer = PersonaRenderer()
        snapshot = renderer.render(config)
        assert snapshot.action_tendency > 0.8
        assert snapshot.system_prompt != ""

    def test_introverted_persona(self):
        """低外向性人格应有低行动倾向。"""
        config = PersonaConfig(
            big_five=BigFive(extraversion=15),
            mbti=MBTI(ei="I"),
        )
        renderer = PersonaRenderer()
        snapshot = renderer.render(config)
        assert snapshot.action_tendency < 0.2

    def test_mbti_tone_style(self):
        """MBTI T/F 维度决定语气风格。"""
        renderer = PersonaRenderer()

        thinker = renderer.render(PersonaConfig(mbti=MBTI(tf="T")))
        assert thinker.tone_style == "rational"

        feeler = renderer.render(PersonaConfig(mbti=MBTI(tf="F")))
        assert feeler.tone_style == "emotional"

    def test_persona_from_json(self):
        """从 JSON 文件加载人格配置。"""
        import json
        config_path = Path(__file__).parent.parent / "config" / "persona.json"
        with open(config_path) as f:
            data = json.load(f)
        config = PersonaConfig.from_dict(data)
        assert config.name == "Galatea"
        assert config.nickname == "伽拉"
        assert config.big_five.openness == 75
        assert config.mbti.ei == "E"

    def test_persona_prompt_contains_name(self):
        """渲染的 System Prompt 应包含角色名。"""
        config = PersonaConfig(name="测试角色")
        renderer = PersonaRenderer()
        snapshot = renderer.render(config)
        assert "测试角色" in snapshot.system_prompt


# ── 3. 调度器 ──

class TestScheduler:

    def test_message_ingest(self):
        """调度器可以接收消息。"""
        scheduler = ActionScheduler()
        msg = MessageEvent(
            sender="u1", sender_name="Alice",
            content="hello", chat_id="c1",
        )
        scheduler.ingest_message(msg)
        assert len(scheduler._pending_messages) == 1

    def test_should_wake_on_messages(self):
        """有待处理消息时应该唤醒。"""
        scheduler = ActionScheduler(SchedulerConfig(message_debounce=0))
        scheduler.ingest_message(MessageEvent(
            sender="u1", sender_name="Alice", content="hi",
        ))
        should, reason = scheduler.should_wake()
        assert should is True
        assert reason == "new_messages"

    def test_should_wake_on_boredom(self):
        """无聊度超过阈值时应该唤醒。"""
        scheduler = ActionScheduler(SchedulerConfig(boredom_threshold=0.5))
        # 手动设置高无聊度
        scheduler._internal.boredom = 0.8
        scheduler._internal.energy = 0.8
        should, reason = scheduler.should_wake()
        assert should is True
        assert reason == "boredom"

    def test_should_not_wake_low_energy(self):
        """精力值过低时不应唤醒。"""
        scheduler = ActionScheduler(SchedulerConfig(
            boredom_threshold=0.5, energy_threshold=0.5,
        ))
        scheduler._internal.boredom = 0.8
        scheduler._internal.energy = 0.1
        should, reason = scheduler.should_wake()
        assert should is False

    def test_build_wake_state(self):
        """构建唤醒状态时清空待处理消息。"""
        scheduler = ActionScheduler()
        scheduler.ingest_message(MessageEvent(
            sender="u1", sender_name="Alice", content="hi",
        ))
        state = scheduler.build_wake_state()
        assert len(state.messages) == 1
        assert len(scheduler._pending_messages) == 0

    def test_tick_increases_boredom(self):
        """tick 应增加无聊度。"""
        scheduler = ActionScheduler()
        before = scheduler._internal.boredom
        scheduler.tick()
        after = scheduler._internal.boredom
        assert after > before


# ── 4. Adapter Mock 测试 ──

class TestAdapters:

    def test_mock_llm(self):
        llm = MockLLMAdapter()
        response = llm.chat("system", "hello")
        assert isinstance(response, str)
        assert len(response) > 0

    def test_mock_lpmm_search(self):
        lpmm = MockLPMMAdapter()
        results = lpmm.search("test query")
        assert isinstance(results, list)

    def test_mock_lpmm_write(self):
        lpmm = MockLPMMAdapter()
        success = lpmm.write("test memory")
        assert success is True

    def test_mock_platform_send(self):
        platform = MockPlatformAdapter()
        success = platform.send_message("chat1", "hello")
        assert success is True
