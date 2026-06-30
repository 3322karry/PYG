"""pyGal Agent 五节点实现。

阶段 3 扩展：
  - decide 节点接入人格行为参数（概率决策 + LLM CoT 评估）
  - act 节点支持 search → reply 循环
  - observe 节点实现循环控制（search 后继续，其他行动终止）
  - 人格参数贯穿整个决策链
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

from .state import (
    PyGalState, AgentPhase, MessageEvent,
    ActionOutput, Observation, InternalState, MAX_LOOPS,
)

if TYPE_CHECKING:
    from ..adapter.llm import LLMAdapter
    from ..adapter.lpmm import LPMMAdapter
    from ..adapter.platform import PlatformAdapter
    from ..persona.renderer import PersonaRenderer
    from ..tools.manager import ToolManager
    from ..reflection.engine import ReflectionEngine


class AgentNodes:
    """五个节点的实现集合。"""

    def __init__(
        self,
        llm: "LLMAdapter | None" = None,
        lpmm: "LPMMAdapter | None" = None,
        platform: "PlatformAdapter | None" = None,
        persona_renderer: "PersonaRenderer | None" = None,
        tool_manager: "ToolManager | None" = None,
        reflection_engine: "ReflectionEngine | None" = None,
    ):
        self.llm = llm
        self.lpmm = lpmm
        self.platform = platform
        self.persona_renderer = persona_renderer
        self.tool_manager = tool_manager
        self.reflection_engine = reflection_engine

    # ── 1. Perceive 感知 ──────────────────────────────

    def perceive(self, state: PyGalState) -> dict:
        """感知节点：解析消息流，构建感知上下文。"""
        if not state.messages:
            context = "（无新消息，定时触发）"
        else:
            parts = []
            for msg in state.messages:
                mention_tag = " [@我]" if msg.is_mention_me else ""
                parts.append(f"[{msg.sender_name}]{mention_tag}: {msg.content}")
            context = "\n".join(parts)

        # 收到消息时降低无聊度
        if state.messages:
            state.internal.on_message(count=len(state.messages))

        state.add_trace("perceive", {
            "message_count": len(state.messages),
            "trigger": state.trigger,
            "boredom": round(state.internal.boredom, 3),
            "energy": round(state.internal.energy, 3),
        })

        return {"perceived_context": context}

    # ── 2. Reflect 反思 ───────────────────────────────

    def reflect(self, state: PyGalState) -> dict:
        """反思节点：理解消息 + 检索记忆 + LLM 推理。"""
        retrieved = []

        # 检索记忆
        if self.lpmm and state.perceived_context and state.perceived_context != "（无新消息，定时触发）":
            try:
                retrieved = self.lpmm.search(
                    query=state.perceived_context,
                    limit=5,
                    chat_id=state.messages[0].chat_id if state.messages else state.chat_id,
                )
            except Exception as e:
                state.add_trace("reflect", {"lpmm_error": str(e)})
                retrieved = []

        # LLM 反思
        reflection = ""
        if self.llm:
            persona_prompt = state.persona.system_prompt
            prompt = (
                f"{persona_prompt}\n\n"
                f"以下是当前对话场景：\n{state.perceived_context}\n\n"
                f"检索到的记忆：\n"
                + "\n".join(f"- {m.get('content', '')}" for m in retrieved)
                + "\n\n请简短反思：当前对话的重点是什么？我应该如何回应？"
            )
            try:
                reflection = self.llm.chat(
                    system="你是虚拟网友的内心独白。简短反思，不要超过100字。",
                    user=prompt,
                )
            except Exception as e:
                state.add_trace("reflect", {"llm_error": str(e)})
                reflection = "（LLM 不可用，使用默认反思）"

        if not reflection:
            if state.messages:
                reflection = f"收到 {len(state.messages)} 条消息，需要决定是否回复。"
            else:
                reflection = f"定时触发，无聊度 {state.internal.boredom:.2f}，考虑是否主动说点什么。"

        state.add_trace("reflect", {
            "memories_retrieved": len(retrieved),
            "reflection_length": len(reflection),
        })

        return {
            "retrieved_memories": retrieved,
            "reflection": reflection,
        }

    # ── 3. Decide 决策 ────────────────────────────────

    def decide(self, state: PyGalState) -> dict:
        """决策节点：基于人格参数 + 内部状态 + 反思选择行动。

        阶段 3 决策逻辑：
          1. 有消息 → 按人格 reply_willingness 概率决定是否回复
          2. 有问句 → 按人格 curiosity_drive 概率决定是否先搜索
          3. 无消息 + 无聊度高 → 按人格 topic_initiative 概率发起话题
          4. 其他 → 潜水

        如果 LLM 可用，在概率通过后调用 LLM 做 CoT 评估。
        """
        persona = state.persona
        action_type = "silent"
        reasoning = ""
        content = ""

        has_messages = bool(state.messages) and state.trigger == "message"
        is_bored = state.internal.boredom >= 0.6

        if has_messages:
            # ── 有消息：决定回复 / 搜索 / 潜水 ──
            # 被 @ 时回复意愿大幅提升
            mentioned = any(m.is_mention_me for m in state.messages)

            if mentioned:
                # 被 @ 了，几乎一定回复
                reply_prob = 0.98
            else:
                # 回复意愿 = 基础回复意愿 + 社交驱动 * 0.3
                reply_prob = persona.reply_willingness * 0.7 + state.internal.social_drive * 0.3

            if random.random() < reply_prob:
                # 决定回复。是否需要先搜索？
                has_question = "?" in state.perceived_context or "？" in state.perceived_context
                if has_question and self._has_tool("search_web", state):
                    # 按好奇心概率决定是否搜索
                    search_prob = persona.curiosity_drive * 0.6
                    if random.random() < search_prob:
                        action_type = "search"
                        # 提取搜索关键词
                        content = state.perceived_context[:80]
                        reasoning = f"检测到问题，好奇心驱动({persona.curiosity_drive:.2f})触发搜索"
                    else:
                        action_type = "reply"
                        reasoning = f"回复意愿({reply_prob:.2f})通过，直接回复"
                else:
                    action_type = "reply"
                    reasoning = f"回复意愿({reply_prob:.2f})通过，直接回复"
            else:
                action_type = "silent"
                reasoning = f"回复意愿({reply_prob:.2f})未通过，选择潜水"

        elif is_bored and state.internal.energy >= 0.2:
            # ── 无消息但无聊：决定是否主动发话题 ──
            topic_prob = persona.topic_initiative * (
                0.5 + state.internal.boredom * 0.5  # 无聊度越高越想说话
            )

            if random.random() < topic_prob:
                action_type = "topic"
                reasoning = (f"无聊度({state.internal.boredom:.2f}) + "
                           f"话题主动性({persona.topic_initiative:.2f}) → 发起话题")
            else:
                action_type = "silent"
                reasoning = (f"话题主动性({topic_prob:.2f})未通过，继续潜水 "
                           f"(无聊度{state.internal.boredom:.2f})")

        else:
            action_type = "silent"
            reasoning = (f"无聊度({state.internal.boredom:.2f})不足/精力"
                       f"({state.internal.energy:.2f})不足，保持潜水")

        # ── LLM CoT 评估（如果可用）──
        if self.llm and action_type != "silent":
            action_type, reasoning = self._llm_evaluate(
                state, action_type, reasoning
            )

        decision = ActionOutput(
            action_type=action_type,
            content=content,
            reasoning=reasoning,
        )

        state.add_trace("decide", {
            "action_type": decision.action_type,
            "reasoning": decision.reasoning,
            "persona_action_tendency": round(persona.action_tendency, 3),
            "persona_reply_willingness": round(persona.reply_willingness, 3),
            "internal_boredom": round(state.internal.boredom, 3),
            "internal_energy": round(state.internal.energy, 3),
        })

        return {"decision": decision}

    # ── 工具辅助 ──

    def _has_tool(self, name: str, state: PyGalState | None = None) -> bool:
        """检查工具是否可用（已注册且已启用）。"""
        if self.tool_manager:
            return name in self.tool_manager.get_enabled_names()
        # 降级：检查 state.active_tools（阶段 1-3 兼容）
        if state:
            return name in state.active_tools
        return False

    def _execute_tool(self, name: str, **kwargs) -> dict:
        """执行工具调用，返回结果 dict。"""
        if self.tool_manager:
            result = self.tool_manager.execute(name, **kwargs)
            return result.to_dict()
        # 降级：返回 Mock 结果
        return {"success": True, "output": f"[mock] {name} executed", "data": {}, "error": ""}

    def _llm_evaluate(
        self, state: PyGalState, proposed_action: str, reasoning: str
    ) -> tuple[str, str]:
        """LLM Chain of Thought 评估：行动是否符合人设和语境。

        Returns:
            (最终行动类型, 评估后的推理)
        """
        persona = state.persona
        try:
            eval_prompt = (
                f"你的人格: {persona.system_prompt[:500]}\n\n"
                f"当前场景:\n{state.perceived_context}\n\n"
                f"你的反思: {state.reflection}\n\n"
                f"你计划采取的行动: {proposed_action}\n"
                f"理由: {reasoning}\n\n"
                f"请评估这个行动是否符合你的人设和当前语境。"
                f"回复格式: [OK] 或 [CHANGE:行动类型]。"
                f"行动类型可选: reply, search, topic, silent"
            )
            result = self.llm.chat(
                system="你是虚拟网友的行动评估器。只输出 [OK] 或 [CHANGE:xxx]。",
                user=eval_prompt,
                max_tokens=50,
            )
            result = result.strip()

            if result.startswith("[CHANGE:"):
                new_action = result.split(":")[1].rstrip("]").strip()
                if new_action in ("reply", "search", "topic", "silent"):
                    return new_action, f"CoT评估修正: {proposed_action} → {new_action}"

            return proposed_action, f"CoT评估通过: {reasoning}"

        except Exception:
            return proposed_action, reasoning

    # ── 4. Act 行动 ───────────────────────────────────

    def act(self, state: PyGalState) -> dict:
        """行动节点：执行选中行为。"""
        decision = state.decision
        if not decision:
            return {"action_result": None}

        action = ActionOutput(
            action_type=decision.action_type,
            reasoning=decision.reasoning,
        )

        if decision.action_type == "silent":
            state.internal.on_action("silent")
            state.add_trace("act", {"action": "silent"})
            return {"action_result": action}

        if decision.action_type == "reply" and self.llm:
            persona_prompt = state.persona.system_prompt
            style_hint = state.persona.speech_style_hint
            try:
                reply_text = self.llm.chat(
                    system=f"你是一个虚拟网友。{persona_prompt}\n说话风格: {style_hint}",
                    user=(
                        f"对话场景：\n{state.perceived_context}\n\n"
                        f"你的反思：{state.reflection}\n\n"
                        f"请以网友的口吻回复（50字以内）。"
                        f"{'不要使用emoji。' if state.persona.emoji_frequency < 0.3 else ''}"
                        f"{'可以适当使用emoji。' if state.persona.emoji_frequency > 0.5 else ''}"
                    ),
                )
                action.content = reply_text

                # 通过平台适配器发送
                if self.platform:
                    chat_id = state.chat_id or (state.messages[0].chat_id if state.messages else "")
                    if chat_id:
                        self.platform.send_message(chat_id, reply_text)

                state.internal.on_action("reply")

            except Exception as e:
                action.content = f"（回复生成失败: {e}）"
                state.add_trace("act", {"error": str(e)})

        elif decision.action_type == "search":
            # 阶段 4: 通过 ToolManager 执行搜索工具
            search_query = decision.content or state.perceived_context
            tool_result = self._execute_tool("search_web", query=search_query)

            action.tool_calls.append({
                "tool": "search_web",
                "query": search_query,
                "result_success": tool_result["success"],
                "result_output": tool_result["output"][:200],
            })
            action.content = tool_result.get("output", f"（搜索: {search_query}）")
            state.internal.on_action("search")
            state.add_trace("act", {
                "tool_call": "search_web",
                "query": search_query,
                "result_success": tool_result["success"],
                "result_preview": tool_result["output"][:150],
            })

        elif decision.action_type == "topic" and self.llm:
            try:
                topic_text = self.llm.chat(
                    system=f"你是一个虚拟网友。{state.persona.system_prompt}\n说话风格: {state.persona.speech_style_hint}",
                    user=(
                        "你现在想主动在群里说点什么，像一个真实网友那样自然地发起话题。"
                        "可以分享一个想法、吐槽一件事、或者问大家一个问题（30字以内）。"
                    ),
                )
                action.content = topic_text

                if self.platform:
                    chat_id = state.chat_id or (state.messages[0].chat_id if state.messages else "")
                    if chat_id:
                        self.platform.send_message(chat_id, topic_text)

                state.internal.on_action("topic")

            except Exception as e:
                action.content = f"（话题生成失败: {e}）"

        state.add_trace("act", {
            "action_type": action.action_type,
            "content_length": len(action.content),
            "content_preview": action.content[:80],
        })

        return {"action_result": action}

    # ── 5. Observe 观察 ───────────────────────────────

    def observe(self, state: PyGalState) -> dict:
        """观察节点：收集行动结果，决定是否继续循环。

        循环规则：
          - search 行动后 → should_continue=True（搜索完需要回复）
          - 其他行动 → should_continue=False
          - 超过 MAX_LOOPS → 强制终止
        """
        observation = Observation(
            success=True,
            feedback="",
            new_messages=[],
        )

        if state.action_result:
            ar_type = state.action_result.action_type

            if ar_type in ("reply", "topic"):
                observation.feedback = f"{ar_type} 完成，精力 -0.15, 无聊度归零"
            elif ar_type == "search":
                observation.feedback = "搜索完成，需要继续回复"
            elif ar_type == "silent":
                observation.feedback = f"潜水中，无聊度 +0.05"

        # ── 循环控制 ──
        state.loop_count += 1

        if state.loop_count >= MAX_LOOPS:
            state.should_continue = False
            observation.feedback += f" (达到最大循环 {MAX_LOOPS}，终止)"
            # 阶段 5: 循环结束时触发反思
            self._maybe_reflect(state)
        elif state.action_result and state.action_result.action_type == "search":
            # search 后继续循环（让 Agent 基于搜索结果回复）
            state.should_continue = True
            observation.feedback += " (继续循环: search → reply)"
        else:
            state.should_continue = False
            # 阶段 5: 非循环行动结束后触发反思
            self._maybe_reflect(state)

        state.add_trace("observe", {
            "energy": round(state.internal.energy, 3),
            "boredom": round(state.internal.boredom, 3),
            "loop_count": state.loop_count,
            "should_continue": state.should_continue,
            "feedback": observation.feedback,
        })

        return {"observation": observation, "should_continue": state.should_continue}

    # ── 反思触发 ──

    def _maybe_reflect(self, state: PyGalState) -> None:
        """阶段 5: Agent 循环结束时触发反思。"""
        if not self.reflection_engine:
            return

        from ..reflection.engine import ReflectionTrigger

        # 深度对话触发（有消息且执行了回复/话题）
        if state.messages and state.action_result:
            if state.action_result.action_type in ("reply", "topic"):
                if self.reflection_engine.should_reflect(state, ReflectionTrigger.DEEP_CONVERSATION):
                    result = self.reflection_engine.reflect(state, ReflectionTrigger.DEEP_CONVERSATION)
                    state.add_trace("reflection", {
                        "trigger": "deep_conversation",
                        "memory_nodes_written": len(result.memory_nodes),
                        "impressions_updated": len(result.impression_updates),
                        "success": result.success,
                        "error": result.error[:100] if result.error else "",
                    })
