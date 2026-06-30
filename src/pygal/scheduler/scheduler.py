"""主动行动调度器 — 消息事件 + 定时器双触发。

阶段 3 实现：
  - 完整的精力值/无聊度衰减/恢复逻辑（接入人格参数）
  - 消息防抖
  - 主动触发频率受人格 action_tendency 调节
  - 异步事件循环
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from ..core.state import PyGalState, MessageEvent, InternalState
from ..persona.model import PersonaSnapshot


@dataclass
class SchedulerConfig:
    """调度器配置。"""
    # 定时检查间隔（秒）
    check_interval: float = 30.0
    # 无聊度阈值（超过则触发主动行为）
    boredom_threshold: float = 0.7
    # 精力值阈值（低于则不触发主动行为）
    energy_threshold: float = 0.2
    # 消息防抖时间（秒）
    message_debounce: float = 1.5
    # 主动行为最小间隔（秒）
    min_action_interval: float = 300.0
    # 单次 tick 间隔（秒）—— 内部状态更新频率
    tick_interval: float = 10.0


class ActionScheduler:
    """主动行动调度器。

    两种触发方式:
      1. 消息事件触发: 收到新消息时，防抖后唤醒 Agent
      2. 定时器触发: 定期检查内部状态（无聊度等），决定是否主动行动

    主动触发频率受人格参数调节:
      - action_tendency 高 → 主动行为间隔短
      - action_tendency 低 → 主动行为间隔长

    调度器本身不执行 Agent 逻辑——它只负责决定「何时唤醒」，
    具体行动由 AgentCore 状态图决定。
    """

    def __init__(
        self,
        config: Optional[SchedulerConfig] = None,
        on_wake: Optional[Callable[[PyGalState], None]] = None,
        persona: Optional[PersonaSnapshot] = None,
    ):
        self.config = config or SchedulerConfig()
        self._on_wake = on_wake
        self._internal = InternalState()
        self._persona = persona
        self._last_action_time: float = 0.0
        self._pending_messages: list[MessageEvent] = []
        self._last_message_time: float = 0.0
        self._running = False
        self._timer_task: Optional[asyncio.Task] = None
        self._debounce_task: Optional[asyncio.Task] = None

    def set_persona(self, persona: PersonaSnapshot) -> None:
        """更新人格快照（影响衰减速率和触发频率）。"""
        self._persona = persona

    @property
    def internal(self) -> InternalState:
        return self._internal

    @property
    def pending_count(self) -> int:
        return len(self._pending_messages)

    def ingest_message(self, message: MessageEvent) -> None:
        """接收一条外部消息。

        由 MaiBot 的消息接收回调调用。
        消息不会立即触发 Agent——而是先缓存，等防抖时间过后再批量触发。
        """
        self._pending_messages.append(message)
        self._internal.on_message(count=1)
        self._last_message_time = time.time()

    def _get_effective_action_interval(self) -> float:
        """根据人格调整主动行为最小间隔。"""
        base = self.config.min_action_interval
        if self._persona:
            # action_tendency 高 → 间隔短（最快 60s）
            # action_tendency 低 → 间隔长（最慢 1800s）
            multiplier = 2.0 - self._persona.action_tendency * 1.5
            return base * multiplier
        return base

    def should_wake(self) -> tuple[bool, str]:
        """判断是否应该唤醒 Agent。

        Returns:
            (是否唤醒, 唤醒原因)
                原因: "new_messages" | "boredom" | ""
        """
        now = time.time()

        # 原因 1: 有待处理消息（防抖时间已过）
        if self._pending_messages:
            debounce_passed = now - self._last_message_time >= self.config.message_debounce
            if debounce_passed:
                return True, "new_messages"

        # 原因 2: 无聊度超过阈值且有精力
        min_interval = self._get_effective_action_interval()
        if (
            self._internal.boredom >= self.config.boredom_threshold
            and self._internal.energy >= self.config.energy_threshold
            and now - self._last_action_time >= min_interval
        ):
            return True, "boredom"

        return False, ""

    def build_wake_state(self, chat_id: str = "default") -> PyGalState:
        """构建唤醒时的初始状态。"""
        has_messages = bool(self._pending_messages)
        reason = "new_messages" if has_messages else "boredom"

        state = PyGalState()
        state.chat_id = chat_id
        state.trigger = "message" if has_messages else "timer"
        state.messages = list(self._pending_messages)
        state.internal = self._internal
        if self._persona:
            state.persona = self._persona

        state.add_trace("scheduler_wake", {
            "reason": reason,
            "pending_count": len(self._pending_messages),
            "energy": round(self._internal.energy, 3),
            "boredom": round(self._internal.boredom, 3),
            "social_drive": round(self._internal.social_drive, 3),
        })

        # 清空待处理消息
        self._pending_messages.clear()
        self._last_action_time = time.time()

        return state

    def tick(self) -> None:
        """时间流逝 —— 更新内部状态。

        阶段 3：调用 InternalState.tick()，传入人格参数调节速率。
        """
        self._internal.tick(persona=self._persona)

    def update_after_action(self, action_type: str) -> None:
        """Agent 执行完行动后，更新调度器状态。"""
        self._internal.on_action(action_type)
        self._last_action_time = time.time()

    async def start(self) -> None:
        """启动定时器循环。"""
        self._running = True
        while self._running:
            self.tick()
            should, reason = self.should_wake()
            if should and self._on_wake:
                state = self.build_wake_state()
                self._on_wake(state)
            await asyncio.sleep(self.config.tick_interval)

    async def stop(self) -> None:
        """停止定时器循环。"""
        self._running = False
        if self._timer_task:
            self._timer_task.cancel()
            self._timer_task = None
        if self._debounce_task:
            self._debounce_task.cancel()
            self._debounce_task = None
