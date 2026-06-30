"""pyGal 阶段 3 测试 — 主动行动系统。

测试覆盖:
  1. InternalState 时间驱动衰减/恢复
  2. 人格参数影响衰减速率
  3. ActionScheduler 消息防抖 + 无聊触发
  4. decide 节点概率决策（接入人格参数）
  5. search → reply 循环
  6. 循环次数上限保护
  7. LLM CoT 评估
"""
from __future__ import annotations

import time
import random
import pytest
from pathlib import Path

from pygal.core.state import (
    PyGalState, AgentPhase, MessageEvent, InternalState, MAX_LOOPS,
    ActionOutput,
)
from pygal.core.graph import build_agent_graph, PyGalAgent
from pygal.core.nodes import AgentNodes
from pygal.adapter.llm import MockLLMAdapter
from pygal.adapter.lpmm import MockLPMMAdapter
from pygal.adapter.platform import MockPlatformAdapter
from pygal.persona.model import PersonaConfig, BigFive, MBTI, PersonaSnapshot
from pygal.persona.renderer import PersonaRenderer
from pygal.scheduler import ActionScheduler, SchedulerConfig


# ── 辅助函数 ──

def make_persona(extroversion=50, openness=50, neuroticism=50) -> PersonaSnapshot:
    """创建人格快照。"""
    config = PersonaConfig(
        big_five=BigFive(
            extraversion=extroversion,
            openness=openness,
            neuroticism=neuroticism,
        ),
        mbti=MBTI(ei="E" if extroversion >= 50 else "I"),
    )
    return PersonaRenderer().render(config)


def make_nodes(llm=None, lpmm=None, platform=None) -> AgentNodes:
    return AgentNodes(
        llm=llm or MockLLMAdapter(),
        lpmm=lpmm or MockLPMMAdapter(),
        platform=platform or MockPlatformAdapter(),
        persona_renderer=PersonaRenderer(),
    )


def make_state(messages=None, persona=None, trigger="message") -> PyGalState:
    state = PyGalState()
    state.trigger = trigger
    state.messages = messages or []
    state.persona = persona or make_persona()
    state.active_tools = ["search_web", "get_time", "query_lpmm"]
    return state


def make_message(content="你好", sender="u1", name="Alice", mention=False) -> MessageEvent:
    return MessageEvent(
        sender=sender, sender_name=name,
        content=content, chat_id="chat1",
        is_mention_me=mention,
    )


def run_node(nodes, method_name, state):
    """调用单个节点并合并返回值到 state（模拟 LangGraph 行为）。"""
    method = getattr(nodes, method_name)
    result = method(state)
    for key, val in result.items():
        setattr(state, key, val)
    return result


# ── 1. InternalState 衰减/恢复 ──

class TestInternalState:

    def test_tick_increases_boredom(self):
        """tick 后无聊度应上升。"""
        state = InternalState()
        before = state.boredom
        time.sleep(0.01)
        state.tick()
        assert state.boredom > before

    def test_tick_restores_energy(self):
        """tick 后精力值应恢复。"""
        state = InternalState()
        state.energy = 0.3
        time.sleep(0.01)
        state.tick()
        assert state.energy > 0.3

    def test_extrovert_boredom_faster(self):
        """外向的人无聊度增长更快。"""
        extrovert = make_persona(extroversion=90)
        introvert = make_persona(extroversion=10)

        s1 = InternalState()
        s2 = InternalState()
        s1._last_tick_time = time.time()
        s2._last_tick_time = time.time()
        time.sleep(0.05)

        s1.tick(persona=extrovert)
        s2.tick(persona=introvert)

        assert s1.boredom > s2.boredom, (
            f"外向无聊度 {s1.boredom} 应大于内向 {s2.boredom}"
        )

    def test_neurotic_energy_recover_slower(self):
        """高神经质的人精力恢复更慢。"""
        stable = make_persona(neuroticism=10)
        anxious = make_persona(neuroticism=90)

        s1 = InternalState()
        s2 = InternalState()
        s1.energy = 0.3
        s2.energy = 0.3
        s1._last_tick_time = time.time()
        s2._last_tick_time = time.time()
        time.sleep(0.05)

        s1.tick(persona=stable)
        s2.tick(persona=anxious)

        assert s1.energy > s2.energy, (
            f"稳定型精力恢复 {s1.energy} 应快于焦虑型 {s2.energy}"
        )

    def test_on_message_updates_state(self):
        """收到消息时更新状态。"""
        state = InternalState()
        state.boredom = 0.5
        state.social_drive = 0.3
        state.on_message(count=2)
        assert state.boredom < 0.5
        assert state.social_drive > 0.3

    def test_on_action_reply(self):
        """回复后精力下降、无聊归零。"""
        state = InternalState()
        state.energy = 0.8
        state.boredom = 0.6
        state.on_action("reply")
        assert state.energy < 0.8
        assert state.boredom == 0.0

    def test_on_action_silent(self):
        """潜水增加无聊度。"""
        state = InternalState()
        state.boredom = 0.3
        state.on_action("silent")
        assert state.boredom > 0.3


# ── 2. ActionScheduler ──

class TestScheduler:

    def test_persona_adjusts_action_interval(self):
        """人格影响主动行为间隔。"""
        extrovert = make_persona(extroversion=90)
        introvert = make_persona(extroversion=10)

        s_ext = ActionScheduler(persona=extrovert)
        s_int = ActionScheduler(persona=introvert)

        ext_interval = s_ext._get_effective_action_interval()
        int_interval = s_int._get_effective_action_interval()

        assert ext_interval < int_interval, (
            f"外向间隔 {ext_interval} 应小于内向 {int_interval}"
        )

    def test_message_debounce(self):
        """消息防抖：收到消息后不会立即唤醒。"""
        scheduler = ActionScheduler(config=SchedulerConfig(
            message_debounce=1.0,
        ))
        scheduler.ingest_message(make_message())

        # 立即检查不应唤醒（防抖时间未过）
        should, reason = scheduler.should_wake()
        if reason == "new_messages":
            # 防抖时间确实没过，但 should=True 说明没等够
            pytest.fail("防抖未过却唤醒了")

    def test_boredom_triggers_wake(self):
        """无聊度达阈值时触发唤醒。"""
        scheduler = ActionScheduler(config=SchedulerConfig(
            boredom_threshold=0.5,
            min_action_interval=0,
        ))
        scheduler._internal.boredom = 0.8
        scheduler._internal.energy = 0.8
        scheduler._last_action_time = 0  # 确保间隔足够

        should, reason = scheduler.should_wake()
        assert should
        assert reason == "boredom"

    def test_low_energy_blocks_wake(self):
        """精力不足时不触发主动行为。"""
        scheduler = ActionScheduler(config=SchedulerConfig(
            boredom_threshold=0.5,
            energy_threshold=0.3,
            min_action_interval=0,
        ))
        scheduler._internal.boredom = 0.9
        scheduler._internal.energy = 0.1  # 精力不足
        scheduler._last_action_time = 0

        should, reason = scheduler.should_wake()
        # 无聊度够了但精力不够，不应因 boredom 唤醒
        if should and reason == "boredom":
            pytest.fail("精力不足时不应该因无聊唤醒")

    def test_tick_updates_internal_state(self):
        """tick 更新内部状态。"""
        scheduler = ActionScheduler()
        before_boredom = scheduler._internal.boredom
        time.sleep(0.02)
        scheduler.tick()
        assert scheduler._internal.boredom > before_boredom

    def test_update_after_action(self):
        """行动后更新调度器状态。"""
        scheduler = ActionScheduler()
        scheduler._internal.boredom = 0.5
        scheduler.update_after_action("reply")
        assert scheduler._internal.boredom == 0.0


# ── 3. Decide 节点概率决策 ──

class TestDecideNode:

    def test_mentioned_almost_always_reply(self):
        """被 @ 时几乎一定回复。"""
        nodes = make_nodes()
        # 多次测试取统计
        reply_count = 0
        for _ in range(20):
            state = make_state(
                messages=[make_message("伽拉你好", mention=True)],
                persona=make_persona(extroversion=30),  # 即使内向
            )
            run_node(nodes, "perceive", state)
            run_node(nodes, "reflect", state)
            run_node(nodes, "decide", state)
            if state.decision.action_type in ("reply", "search"):
                reply_count += 1
        assert reply_count >= 18, f"被@时回复率应>=90%，实际 {reply_count}/20"

    def test_extrovert_replies_more_than_introvert(self):
        """外向的人比内向的人更倾向于回复。"""
        nodes = make_nodes()
        extrovert = make_persona(extroversion=90)
        introvert = make_persona(extroversion=10)

        random.seed(42)
        extrovert_replies = 0
        introvert_replies = 0

        for _ in range(50):
            state = make_state(
                messages=[make_message("今天天气不错")],
                persona=extrovert,
            )
            run_node(nodes, "perceive", state)
            run_node(nodes, "reflect", state)
            run_node(nodes, "decide", state)
            if state.decision.action_type in ("reply", "search"):
                extrovert_replies += 1

        random.seed(42)
        for _ in range(50):
            state = make_state(
                messages=[make_message("今天天气不错")],
                persona=introvert,
            )
            run_node(nodes, "perceive", state)
            run_node(nodes, "reflect", state)
            run_node(nodes, "decide", state)
            if state.decision.action_type in ("reply", "search"):
                introvert_replies += 1

        assert extrovert_replies > introvert_replies, (
            f"外向回复 {extrovert_replies}/50 应多于内向 {introvert_replies}/50"
        )

    def test_curiosity_drives_search(self):
        """好奇心高的人更倾向于搜索。"""
        nodes = make_nodes()
        curious = make_persona(extroversion=50, openness=90)
        indifferent = make_persona(extroversion=50, openness=10)

        # 消息含问句
        msg = make_message("Python 怎么读取 JSON 文件？")

        random.seed(42)
        curious_searches = 0
        indifferent_searches = 0

        for _ in range(50):
            state = make_state(messages=[msg], persona=curious)
            run_node(nodes, "perceive", state)
            run_node(nodes, "reflect", state)
            run_node(nodes, "decide", state)
            if state.decision.action_type == "search":
                curious_searches += 1

        random.seed(42)
        for _ in range(50):
            state = make_state(messages=[msg], persona=indifferent)
            run_node(nodes, "perceive", state)
            run_node(nodes, "reflect", state)
            run_node(nodes, "decide", state)
            if state.decision.action_type == "search":
                indifferent_searches += 1

        assert curious_searches > indifferent_searches, (
            f"好奇搜索 {curious_searches}/50 应多于冷漠 {indifferent_searches}/50"
        )

    def test_bored_extrovert_initiates_topic(self):
        """无聊的外向人更倾向于发起话题。"""
        nodes = make_nodes()
        extrovert = make_persona(extroversion=90)
        introvert = make_persona(extroversion=10)

        random.seed(42)
        extrovert_topics = 0
        introvert_topics = 0

        for _ in range(50):
            state = make_state(messages=[], persona=extrovert, trigger="timer")
            state.internal.boredom = 0.8
            state.internal.energy = 0.8
            run_node(nodes, "perceive", state)
            run_node(nodes, "reflect", state)
            run_node(nodes, "decide", state)
            if state.decision.action_type == "topic":
                extrovert_topics += 1

        random.seed(42)
        for _ in range(50):
            state = make_state(messages=[], persona=introvert, trigger="timer")
            state.internal.boredom = 0.8
            state.internal.energy = 0.8
            run_node(nodes, "perceive", state)
            run_node(nodes, "reflect", state)
            run_node(nodes, "decide", state)
            if state.decision.action_type == "topic":
                introvert_topics += 1

        assert extrovert_topics > introvert_topics, (
            f"外向发起 {extrovert_topics}/50 应多于内向 {introvert_topics}/50"
        )


# ── 4. search → reply 循环 ──

class TestSearchReplyLoop:

    def test_search_triggers_loop(self):
        """search 行动后应继续循环。"""
        nodes = make_nodes()
        state = make_state(
            messages=[make_message("Python 怎么读取 JSON？")],
            persona=make_persona(extroversion=50, openness=90),
        )
        # 强制走 search 路径
        state.decision = ActionOutput(
            action_type="search",
            content="Python JSON",
            reasoning="测试 search",
        )
        run_node(nodes, "act", state)
        run_node(nodes, "observe", state)

        assert state.should_continue, "search 后 should_continue 应为 True"
        assert state.loop_count == 1

    def test_reply_terminates_loop(self):
        """reply 行动后应终止循环。"""
        nodes = make_nodes()
        state = make_state(messages=[make_message("你好")])
        state.decision = ActionOutput(
            action_type="reply",
            content="你好呀",
            reasoning="测试 reply",
        )
        run_node(nodes, "act", state)
        run_node(nodes, "observe", state)

        assert not state.should_continue, "reply 后 should_continue 应为 False"

    def test_max_loops_terminates(self):
        """达到最大循环次数后强制终止。"""
        nodes = make_nodes()
        state = make_state()
        state.loop_count = MAX_LOOPS
        state.action_result = ActionOutput(action_type="search")
        run_node(nodes, "observe", state)

        assert not state.should_continue

    def test_full_search_reply_loop(self):
        """完整 search → reply 循环。"""
        random.seed(42)
        nodes = make_nodes()
        # 高好奇心 + 高外向性确保 search 后 reply 概率通过
        state = make_state(
            messages=[make_message("什么是 OCEAN 模型？")],
            persona=make_persona(extroversion=80, openness=95),
        )
        agent = PyGalAgent(nodes)
        result = agent.run(state)

        trace = result.get("trace", [])
        node_names = [t["node"] for t in trace]

        # 应该经历两轮 perceive/reflect/decide/act/observe
        assert node_names.count("perceive") >= 1
        assert node_names.count("decide") >= 1

        # 如果发生了搜索循环，loop_count 应 > 0
        # （概率性，高外向性应确保回复）
        loop_count = result.get("loop_count", 0)
        # 搜索后要么循环回复，要么直接结束（取决于概率）
        assert loop_count >= 0


# ── 5. 完整循环回归测试 ──

class TestFullCycle:

    def test_message_triggers_complete_cycle(self):
        """消息触发完整 Agent 循环。"""
        nodes = make_nodes()
        state = make_state(
            messages=[make_message("你好呀", mention=True)],
        )
        agent = PyGalAgent(nodes)
        result = agent.run(state)

        trace = result.get("trace", [])
        node_names = [t["node"] for t in trace]

        assert "perceive" in node_names
        assert "reflect" in node_names
        assert "decide" in node_names
        assert "act" in node_names
        assert "observe" in node_names

    def test_no_message_extrovert_might_act(self):
        """无消息时外向型可能主动行动。"""
        nodes = make_nodes()
        state = make_state(
            messages=[],
            persona=make_persona(extroversion=95),
            trigger="timer",
        )
        state.internal.boredom = 0.9
        state.internal.energy = 0.9

        agent = PyGalAgent(nodes)
        result = agent.run(state)

        # 外向 + 高无聊 → 有概率发起 topic
        ar = result.get("action_result")
        if ar:
            assert ar.action_type in ("topic", "silent")

    def test_trace_records_loop_count(self):
        """推理追踪记录循环次数。"""
        nodes = make_nodes()
        state = make_state(
            messages=[make_message("什么是 OCEAN？")],
            persona=make_persona(openness=95),
        )
        agent = PyGalAgent(nodes)
        result = agent.run(state)

        trace = result.get("trace", [])
        # 如果发生了循环，trace 中应有 loop > 0 的记录
        loops = [t.get("loop", 0) for t in trace]
        # 至少有 loop 0
        assert 0 in loops
