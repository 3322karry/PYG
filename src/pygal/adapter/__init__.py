"""pyGal 适配层 — 将 MaiBot 底层能力封装为 Agent 可用接口。

三个适配器:
  - LLMAdapter:      封装 MaiBot 的 llm_models 多模型调用
  - LPMMAdapter:     封装 A-Memorix 记忆检索/写入
  - PlatformAdapter: 封装 PlatformIO 消息收发

阶段 1 定义接口 + Mock 实现，后续阶段对接 MaiBot 真实接口。
"""

from .llm import LLMAdapter, MockLLMAdapter
from .lpmm import LPMMAdapter, MockLPMMAdapter
from .platform import PlatformAdapter, MockPlatformAdapter

__all__ = [
    "LLMAdapter", "MockLLMAdapter",
    "LPMMAdapter", "MockLPMMAdapter",
    "PlatformAdapter", "MockPlatformAdapter",
]
