"""LLM 适配器 — 封装 MaiBot 的 llm_models 多模型调用。

MaiBot 原生 LLM 层位于 `src/llm_models/`，支持多模型切换、重试、超时。
本适配器将其封装为统一的 chat() 接口，供 AgentNodes 调用。

阶段 1 提供 MockLLMAdapter 用于独立测试。
阶段 3+ 实现真实 MaiBotLLMAdapter 对接。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional


class LLMAdapter(ABC):
    """LLM 调用的抽象接口。"""

    @abstractmethod
    def chat(
        self,
        system: str,
        user: str,
        temperature: float = 0.8,
        max_tokens: int = 512,
    ) -> str:
        """同步调用 LLM 生成回复。

        Args:
            system: System Prompt
            user: User Prompt
            temperature: 温度参数
            max_tokens: 最大生成 token 数

        Returns:
            LLM 生成的文本
        """
        ...

    @abstractmethod
    async def chat_async(
        self,
        system: str,
        user: str,
        temperature: float = 0.8,
        max_tokens: int = 512,
    ) -> str:
        """异步调用 LLM。"""
        ...


class MockLLMAdapter(LLMAdapter):
    """Mock LLM 适配器 — 用于独立测试，不依赖真实 LLM。"""

    def __init__(self, responses: Optional[list[str]] = None):
        """初始化 Mock LLM。

        Args:
            responses: 预设回复列表。如果为 None 则生成默认回复。
        """
        self._responses = responses or []
        self._call_count = 0
        self._call_history: list[dict] = []

    def chat(
        self,
        system: str,
        user: str,
        temperature: float = 0.8,
        max_tokens: int = 512,
    ) -> str:
        self._call_count += 1
        self._call_history.append({
            "system": system[:100],
            "user": user[:100],
            "temperature": temperature,
        })

        if self._responses:
            idx = (self._call_count - 1) % len(self._responses)
            return self._responses[idx]

        # 默认回复
        return "嗯嗯，说得好呀~"

    async def chat_async(
        self,
        system: str,
        user: str,
        temperature: float = 0.8,
        max_tokens: int = 512,
    ) -> str:
        return self.chat(system, user, temperature, max_tokens)

    @property
    def call_count(self) -> int:
        return self._call_count

    @property
    def call_history(self) -> list[dict]:
        return list(self._call_history)


class MaiBotLLMAdapter(LLMAdapter):
    """真实 MaiBot LLM 适配器 — 对接 MaiBot 的 llm_models 层。

    通过 MaiBot 的 LLMUtils (utils_model.py) 调用 LLM:
      1. 使用 task_name 从 bot_config.toml 读取模型配置
      2. 调用 generate_response_with_message_async 生成回复
      3. 自动处理重试/超时/模型切换
    """

    def __init__(self, task_name: str = "chat", request_type: str = "chat"):
        """
        Args:
            task_name: MaiBot 配置中的任务名（对应 bot_config.toml 中的模型任务配置）
            request_type: 请求类型
        """
        self._task_name = task_name
        self._request_type = request_type
        self._utils = None
        self._available = False
        self._init_tried = False

    def _ensure_init(self):
        """延迟初始化 LLMUtils（需要 MaiBot 运行时环境）。"""
        if self._init_tried:
            return
        self._init_tried = True
        try:
            from llm_models.utils_model import LLMUtils
            self._utils = LLMUtils(
                task_name=self._task_name,
                request_type=self._request_type,
            )
            self._available = True
        except Exception as e:
            import logging
            logging.getLogger("pygal").warning(f"MaiBotLLMAdapter 初始化失败: {e}")
            self._available = False

    def chat(
        self,
        system: str,
        user: str,
        temperature: float = 0.8,
        max_tokens: int = 512,
    ) -> str:
        """同步调用 LLM（内部用 asyncio 包装异步调用）。"""
        import asyncio
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.chat_async(system, user, temperature, max_tokens))
        else:
            # 在已有事件循环中，用 ensure_future
            future = asyncio.ensure_future(self.chat_async(system, user, temperature, max_tokens))
            return loop.run_until_complete(future)

    async def chat_async(
        self,
        system: str,
        user: str,
        temperature: float = 0.8,
        max_tokens: int = 512,
    ) -> str:
        """异步调用 LLM。"""
        import asyncio
        self._ensure_init()
        if not self._available:
            # 降级到 Mock
            return MockLLMAdapter().chat(system, user, temperature, max_tokens)

        try:
            from llm_models.payload_content import Message

            messages = [
                Message(role="system", content=system),
                Message(role="user", content=user),
            ]

            result = await self._utils.generate_response_with_message_async(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            if result and result.completion:
                return result.completion
            return ""

        except Exception as e:
            import logging
            logging.getLogger("pygal").error(f"LLM 调用失败: {e}")
            return f"（LLM 调用失败: {e}）"
