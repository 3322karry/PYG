"""pyGal Agent 状态图构建。

使用 LangGraph StateGraph 构建五节点循环图：

    START → perceive → reflect → decide → act → observe → END
       ↑                                          │
       └──────────── should_continue ─────────────┘

阶段 3 实现循环逻辑：
  - search 行动后 observe 设 should_continue=True，路由回 perceive
  - 循环次数上限 MAX_LOOPS 防止无限循环
"""

from __future__ import annotations

from typing import Optional

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from .state import PyGalState, AgentPhase
from .nodes import AgentNodes


def build_agent_graph(
    nodes: AgentNodes,
    use_checkpoint: bool = False,
) -> StateGraph:
    """构建 pyGal Agent 状态图。

    Args:
        nodes: 五节点实现集合
        use_checkpoint: 是否启用 LangGraph MemorySaver（供 WebUI 回放）

    Returns:
        编译后的 LangGraph 可执行图
    """
    graph = StateGraph(PyGalState)

    # ── 添加节点 ──
    graph.add_node(AgentPhase.PERCEIVE.value, nodes.perceive)
    graph.add_node(AgentPhase.REFLECT.value, nodes.reflect)
    graph.add_node(AgentPhase.DECIDE.value, nodes.decide)
    graph.add_node(AgentPhase.ACT.value, nodes.act)
    graph.add_node(AgentPhase.OBSERVE.value, nodes.observe)

    # ── 线性边 ──
    graph.add_edge(START, AgentPhase.PERCEIVE.value)
    graph.add_edge(AgentPhase.PERCEIVE.value, AgentPhase.REFLECT.value)
    graph.add_edge(AgentPhase.REFLECT.value, AgentPhase.DECIDE.value)
    graph.add_edge(AgentPhase.DECIDE.value, AgentPhase.ACT.value)
    graph.add_edge(AgentPhase.ACT.value, AgentPhase.OBSERVE.value)

    # ── 条件边：observe → perceive（循环）或 END ──
    def _observe_router(state: PyGalState) -> str:
        """observe 后的路由：继续循环或终止。"""
        if state.should_continue and state.loop_count < 10:
            return AgentPhase.PERCEIVE.value
        return END

    graph.add_conditional_edges(
        AgentPhase.OBSERVE.value,
        _observe_router,
        {
            AgentPhase.PERCEIVE.value: AgentPhase.PERCEIVE.value,
            END: END,
        },
    )

    # ── 编译 ──
    checkpointer = MemorySaver() if use_checkpoint else None
    return graph.compile(checkpointer=checkpointer)


class PyGalAgent:
    """pyGal Agent 封装 — 对外暴露的统一接口。

    使用方式:
        agent = PyGalAgent(nodes)
        result = agent.run(initial_state)
    """

    def __init__(
        self,
        nodes: AgentNodes,
        use_checkpoint: bool = False,
    ):
        self.nodes = nodes
        self.graph = build_agent_graph(nodes, use_checkpoint=use_checkpoint)
        self._thread_id: Optional[str] = None

    def run(
        self,
        state: PyGalState,
        thread_id: str = "default",
    ) -> dict:
        """执行一次完整的 Agent 循环。

        Args:
            state: 初始状态
            thread_id: 会话线程 ID（用于 checkpoint 隔离）

        Returns:
            执行完成后的最终状态（dict 形式）
        """
        config = {"configurable": {"thread_id": thread_id}}
        result = self.graph.invoke(state, config=config)
        return result

    def get_trace(self, result: dict) -> list[dict]:
        """获取推理链追踪（供 WebUI agent_trace 使用）。"""
        return result.get("trace", [])

    @property
    def thread_id(self) -> str:
        return self._thread_id or "default"
