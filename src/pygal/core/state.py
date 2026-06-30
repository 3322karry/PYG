"""pyGal Agent 状态定义。

阶段 3 扩展：
  - InternalState 加入时间驱动衰减/恢复逻辑
  - 加入 loop_count 上限控制
"""

from __future__ import annotations

import time
from enum import Enum
from typing import Any, Optional
from dataclasses import dataclass, field

# 统一使用 persona 模块的 PersonaSnapshot
from ..persona.model import PersonaSnapshot


class AgentPhase(str, Enum):
    """Agent 状态图的五个阶段。"""
    PERCEIVE = "perceive"    # 感知：接收消息/事件
    REFLECT = "reflect"      # 反思：理解 + 记忆检索
    DECIDE = "decide"        # 决策：选择行动（回复/搜索/主动话题/潜水）
    ACT = "act"              # 行动：执行选中行为
    OBSERVE = "observe"      # 观察：收集行动结果


# Agent 循环最大次数（防止无限循环）
MAX_LOOPS = 5


class InternalState:
    """Agent 内部状态（精力值、无聊度等），随时间衰减/恢复。

    阶段 3 实现：
      - tick() 方法：根据时间差更新状态
      - 人格参数影响衰减/恢复速率
      - 消息事件影响社交驱动
    """

    def __init__(self):
        self.energy: float = 1.0          # 精力值 0~1，低于阈值时减少主动行为
        self.boredom: float = 0.0         # 无聊度 0~1，高于阈值时触发主动行为
        self.social_drive: float = 0.5    # 社交驱动 0~1，影响回复意愿
        self.last_active_time: float = time.time()  # 上次活跃时间戳
        self._last_tick_time: float = time.time()   # 上次 tick 时间

    def tick(self, persona: Optional[PersonaSnapshot] = None) -> None:
        """时间流逝 —— 根据时间差更新内部状态。

        衰减/恢复速率受人格参数影响：
          - 高 extraversion → 无聊度增长更快（外向的人更容易无聊）
          - 高 neuroticism → 精力恢复更慢（敏感的人更容易累）
          - 高 conscientiousness → 精力恢复更稳定
          - 高 social_drive → 社交驱动衰减更慢

        Args:
            persona: 当前人格快照，用于调节速率
        """
        now = time.time()
        dt = now - self._last_tick_time
        self._last_tick_time = now

        if dt <= 0:
            return

        # ── 速率参数（受人格影响）──
        if persona:
            # 外向的人无聊得更快
            boredom_rate = 0.008 + persona.action_tendency * 0.012
            # 神经质高的人精力恢复慢
            energy_recover = 0.006 - persona.emotional_volatility * 0.003
            energy_recover = max(0.002, energy_recover)
            # 社交驱动衰减率
            social_decay = 0.003
        else:
            boredom_rate = 0.01
            energy_recover = 0.005
            social_decay = 0.003

        # 按时间差缩放（以 60 秒为 1 标准单位）
        scale = dt / 60.0

        self.boredom = min(1.0, self.boredom + boredom_rate * scale)
        self.energy = min(1.0, self.energy + energy_recover * scale)
        self.social_drive = max(0.0, self.social_drive - social_decay * scale)

    def on_message(self, count: int = 1) -> None:
        """收到消息时更新状态。"""
        self.social_drive = min(1.0, self.social_drive + 0.1 * count)
        self.boredom = max(0.0, self.boredom - 0.05 * count)

    def on_action(self, action_type: str) -> None:
        """执行行动后更新状态。"""
        self.last_active_time = time.time()
        if action_type in ("reply", "topic"):
            self.energy = max(0.05, self.energy - 0.15)
            self.boredom = 0.0
        elif action_type == "search":
            self.energy = max(0.05, self.energy - 0.05)
        elif action_type == "silent":
            self.boredom = min(1.0, self.boredom + 0.05)

    def to_dict(self) -> dict[str, Any]:
        return {
            "energy": round(self.energy, 3),
            "boredom": round(self.boredom, 3),
            "social_drive": round(self.social_drive, 3),
            "last_active_time": self.last_active_time,
        }

    def __repr__(self) -> str:
        return (f"InternalState(energy={self.energy:.2f}, "
                f"boredom={self.boredom:.2f}, "
                f"social={self.social_drive:.2f})")


@dataclass
class MessageEvent:
    """单条消息事件。"""
    sender: str               # 发送者 ID
    sender_name: str          # 发送者显示名
    content: str              # 消息内容
    platform: str = ""        # 平台标识
    chat_id: str = ""         # 会话 ID
    timestamp: float = 0.0    # 时间戳
    is_mention_me: bool = False  # 是否 @ 了 AI


@dataclass
class ActionOutput:
    """Agent 的行动输出。"""
    action_type: str          # "reply" | "search" | "topic" | "silent"
    content: str = ""         # 输出内容（回复文本/搜索关键词/话题）
    tool_calls: list[dict] = field(default_factory=list)  # 工具调用记录
    reasoning: str = ""       # 推理过程摘要


@dataclass
class Observation:
    """行动后的观察结果。"""
    success: bool = True
    feedback: str = ""        # 行动反馈（如搜索结果、发送状态等）
    new_messages: list[MessageEvent] = field(default_factory=list)


@dataclass
class PyGalState:
    """LangGraph 状态图的共享状态。

    每个节点接收 PyGalState，修改后返回给下一个节点。
    """
    # ── 输入 ──
    trigger: str = "message"  # "message" | "timer" | "internal"
    chat_id: str = ""
    messages: list[MessageEvent] = field(default_factory=list)

    # ── 内部状态 ──
    internal: InternalState = field(default_factory=InternalState)
    persona: PersonaSnapshot = field(default_factory=PersonaSnapshot)
    active_tools: list[str] = field(default_factory=lambda: ["search_web", "get_time", "query_lpmm"])

    # ── 处理中间结果 ──
    perceived_context: str = ""
    retrieved_memories: list[dict] = field(default_factory=list)
    reflection: str = ""
    decision: Optional[ActionOutput] = None

    # ── 输出 ──
    action_result: Optional[ActionOutput] = None
    observation: Optional[Observation] = None

    # ── 推理链追踪 ──
    trace: list[dict] = field(default_factory=list)

    # ── 控制 ──
    loop_count: int = 0
    should_continue: bool = True

    def add_trace(self, node: str, data: dict) -> None:
        """记录一条推理追踪日志。"""
        self.trace.append({
            "node": node,
            "loop": self.loop_count,
            "data": data,
        })
