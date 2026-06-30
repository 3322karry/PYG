"""反思引擎 — 定时/事件触发，提炼短期记忆为长期记忆。

反思流程:
  1. 收集短期记忆（本次对话的感知/反思/行动记录）
  2. 调用 LLM 执行反思任务:
     - "有什么值得记住的？"
     - "我对用户的印象有何改变？"
  3. 解析 LLM 输出，结构化为 MemoryNode + ImpressionUpdate
  4. 主动写入 LPMM
  5. 记录反思日志（供 WebUI agent_trace 展示）

触发条件:
  - 深度对话结束（消息数 >= threshold）
  - 定时触发（每隔 N 轮 Agent 循环）
  - 外部事件（如用户下线、特殊关键词）
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Optional

from .models import MemoryNode, ImpressionUpdate

if TYPE_CHECKING:
    from ..adapter.llm import LLMAdapter
    from ..adapter.lpmm import LPMMAdapter
    from ..core.state import PyGalState


class ReflectionTrigger(str, Enum):
    """反思触发条件。"""
    DEEP_CONVERSATION = "deep_conversation"  # 深度对话结束
    PERIODIC = "periodic"                    # 定时触发
    EVENT = "event"                          # 特殊事件


@dataclass
class ReflectionResult:
    """反思执行结果。"""
    triggered_by: str = ""
    memory_nodes: list[MemoryNode] = field(default_factory=list)
    impression_updates: list[ImpressionUpdate] = field(default_factory=list)
    raw_llm_output: str = ""
    success: bool = False
    error: str = ""
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "triggered_by": self.triggered_by,
            "memory_nodes": [n.to_dict() for n in self.memory_nodes],
            "impression_updates": [i.to_dict() for i in self.impression_updates],
            "raw_llm_output": self.raw_llm_output[:500],
            "success": self.success,
            "error": self.error,
            "timestamp": self.timestamp,
        }


class ReflectionEngine:
    """反思引擎。

    依赖注入:
      - llm:  LLM 适配器（执行反思推理）
      - lpmm: LPMM 适配器（写入记忆和印象）

    使用方式:
      engine = ReflectionEngine(llm, lpmm)
      result = engine.reflect(state, trigger=ReflectionTrigger.DEEP_CONVERSATION)
    """

    # 触发阈值
    DEEP_CONVERSATION_THRESHOLD = 1  # 消息数 >= 1 即可触发深度对话反思
    PERIODIC_INTERVAL = 10           # 每 10 轮 Agent 循环触发一次
    REFLECTION_COOLDOWN = 30.0       # 反思冷却时间（秒），防止频繁触发

    def __init__(
        self,
        llm: "LPMMAdapter | None" = None,
        lpmm: "LPMMAdapter | None" = None,
    ):
        self.llm = llm
        self.lpmm = lpmm
        self._history: list[ReflectionResult] = []
        self._loop_counter: int = 0
        self._last_reflection_time: float = 0.0

    @property
    def history(self) -> list[ReflectionResult]:
        """反思历史记录（供 WebUI 展示）。"""
        return self._history

    def should_reflect(
        self,
        state: "PyGalState",
        trigger: ReflectionTrigger,
    ) -> bool:
        """判断是否应该触发反思。"""
        import time as _time
        # 冷却检查
        if self._last_reflection_time > 0:
            if _time.time() - self._last_reflection_time < self.REFLECTION_COOLDOWN:
                return False

        if trigger == ReflectionTrigger.DEEP_CONVERSATION:
            return len(state.messages) >= self.DEEP_CONVERSATION_THRESHOLD
        elif trigger == ReflectionTrigger.PERIODIC:
            self._loop_counter += 1
            return self._loop_counter >= self.PERIODIC_INTERVAL
        elif trigger == ReflectionTrigger.EVENT:
            return True
        return False

    def reflect(
        self,
        state: "PyGalState",
        trigger: ReflectionTrigger = ReflectionTrigger.DEEP_CONVERSATION,
    ) -> ReflectionResult:
        """执行反思流程。

        Args:
            state: 当前 Agent 状态
            trigger: 触发条件

        Returns:
            ReflectionResult 包含记忆节点和印象更新
        """
        result = ReflectionResult(triggered_by=trigger.value)

        # 1. 收集短期记忆
        short_term = self._collect_short_term_memory(state)

        # 2. LLM 反思
        if self.llm:
            try:
                llm_output = self._llm_reflect(state, short_term)
                result.raw_llm_output = llm_output

                # 3. 解析 LLM 输出
                nodes, impressions = self._parse_llm_output(llm_output, state)
                result.memory_nodes = nodes
                result.impression_updates = impressions

            except Exception as e:
                result.error = f"LLM 反思失败: {e}"
                result.success = False
                self._history.append(result)
                return result
        else:
            # 无 LLM 时使用 Mock 反思
            nodes, impressions = self._mock_reflect(state)
            result.memory_nodes = nodes
            result.impression_updates = impressions
            result.raw_llm_output = "[mock] 反思完成"

        # 4. 主动写入 LPMM
        if self.lpmm:
            self._write_to_lpmm(result, state)

        # 5. 记录历史
        result.success = True
        import time as _time
        self._last_reflection_time = _time.time()
        self._history.append(result)

        # 重置计数器
        if trigger == ReflectionTrigger.PERIODIC:
            self._loop_counter = 0

        return result

    def _collect_short_term_memory(self, state: "PyGalState") -> str:
        """收集短期记忆（本次对话的摘要）。"""
        parts = []
        for msg in state.messages:
            mention = " [@我]" if msg.is_mention_me else ""
            parts.append(f"[{msg.sender_name}]{mention}: {msg.content}")

        if state.reflection:
            parts.append(f"[内心反思]: {state.reflection}")

        if state.action_result:
            ar = state.action_result
            parts.append(f"[我的行动]: {ar.action_type} - {ar.content[:100]}")

        return "\n".join(parts) or "（无短期记忆）"

    def _llm_reflect(self, state: "PyGalState", short_term: str) -> str:
        """调用 LLM 执行反思。"""
        system = (
            "你是虚拟网友的反思系统。请分析最近的对话，回答两个问题:\n"
            "1. 有什么值得长期记住的？（事实、偏好、重要事件）\n"
            "2. 对参与对话的用户的印象有何改变？\n\n"
            "请用 JSON 格式输出:\n"
            '{"memories": [{"content": "...", "type": "fact|preference|event|emotion", "importance": 0.0-1.0}], '
            '"impressions": [{"person_id": "...", "person_name": "...", "updates": {"kindness": 1}, "summary": "..."}]}'
        )

        user = (
            f"人格背景:\n{state.persona.system_prompt[:300]}\n\n"
            f"近期对话:\n{short_term}\n\n"
            f"请反思并输出 JSON。"
        )

        return self.llm.chat(system=system, user=user, temperature=0.3)

    def _parse_llm_output(
        self,
        output: str,
        state: "PyGalState",
    ) -> tuple[list[MemoryNode], list[ImpressionUpdate]]:
        """解析 LLM 的 JSON 输出为结构化记忆和印象。"""
        nodes: list[MemoryNode] = []
        impressions: list[ImpressionUpdate] = []

        # 尝试提取 JSON
        json_str = output
        if "```json" in output:
            start = output.index("```json") + 7
            end = output.index("```", start)
            json_str = output[start:end]
        elif "```" in output:
            start = output.index("```") + 3
            end = output.index("```", start)
            json_str = output[start:end]

        try:
            data = json.loads(json_str)
        except (json.JSONDecodeError, ValueError):
            # JSON 解析失败，把原始输出作为一条记忆
            if output.strip():
                nodes.append(MemoryNode(
                    content=output.strip()[:200],
                    memory_type="fact",
                    importance=0.3,
                    source="reflection",
                ))
            return nodes, impressions

        # 解析记忆
        for mem in data.get("memories", []):
            nodes.append(MemoryNode(
                content=mem.get("content", ""),
                memory_type=mem.get("type", "fact"),
                importance=float(mem.get("importance", 0.5)),
                source="reflection",
            ))

        # 解析印象
        for imp in data.get("impressions", []):
            impressions.append(ImpressionUpdate(
                person_id=imp.get("person_id", ""),
                person_name=imp.get("person_name", ""),
                updates=imp.get("updates", {}),
                summary=imp.get("summary", ""),
            ))

        return nodes, impressions

    def _mock_reflect(
        self,
        state: "PyGalState",
    ) -> tuple[list[MemoryNode], list[ImpressionUpdate]]:
        """无 LLM 时的 Mock 反思。"""
        nodes: list[MemoryNode] = []
        impressions: list[ImpressionUpdate] = []

        for msg in state.messages:
            # 简单规则：消息内容作为记忆
            if len(msg.content) > 10:
                nodes.append(MemoryNode(
                    content=f"{msg.sender_name} 说了: {msg.content[:80]}",
                    memory_type="event",
                    importance=0.4,
                    source="reflection",
                    related_persons=[msg.sender],
                ))

            # 简单印象：有互动就 +1 kindness
            impressions.append(ImpressionUpdate(
                person_id=msg.sender,
                person_name=msg.sender_name,
                updates={"kindness": 1},
                summary=f"{msg.sender_name} 参与了对话",
            ))

        return nodes, impressions

    def _write_to_lpmm(
        self,
        result: ReflectionResult,
        state: "PyGalState",
    ) -> None:
        """将反思结果写入 LPMM。"""
        chat_id = state.chat_id or (state.messages[0].chat_id if state.messages else "")

        # 写入记忆节点
        for node in result.memory_nodes:
            try:
                self.lpmm.write(
                    content=node.content,
                    chat_id=chat_id,
                    metadata=node.to_lpmm_metadata(),
                )
            except Exception as e:
                result.error += f" 写入记忆失败: {e};"

        # 更新印象
        for imp in result.impression_updates:
            try:
                self.lpmm.update_impression(
                    person_id=imp.person_id,
                    impression_updates=imp.to_dict(),
                )
            except Exception as e:
                result.error += f" 更新印象失败: {e};"

    def get_history_summary(self) -> list[dict]:
        """获取反思历史摘要（供 WebUI 展示）。"""
        return [r.to_dict() for r in self._history]
