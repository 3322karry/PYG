"""PlatformIO 适配器 — 封装 MaiBot 的消息收发接口。

MaiBot 的平台层位于 `src/platform_io/`，负责:
  - 接收各平台（QQ/Discord/...）的消息
  - 去重、路由
  - 发送回复消息

本适配器将发送能力封装为统一接口。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional


class PlatformAdapter(ABC):
    """平台消息收发抽象接口。"""

    @abstractmethod
    def send_message(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
    ) -> bool:
        """发送消息到指定会话。

        Args:
            chat_id: 目标会话 ID
            content: 消息内容
            reply_to: 回复的消息 ID（可选）

        Returns:
            是否发送成功
        """
        ...

    @abstractmethod
    async def send_message_async(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
    ) -> bool:
        """异步发送消息。"""
        ...


class MockPlatformAdapter(PlatformAdapter):
    """Mock 平台适配器 — 用于独立测试。"""

    def __init__(self):
        self._sent_messages: list[dict[str, Any]] = []

    def send_message(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
    ) -> bool:
        self._sent_messages.append({
            "chat_id": chat_id,
            "content": content,
            "reply_to": reply_to,
        })
        return True

    async def send_message_async(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
    ) -> bool:
        return self.send_message(chat_id, content, reply_to)

    @property
    def sent_messages(self) -> list[dict[str, Any]]:
        return list(self._sent_messages)

    @property
    def last_message(self) -> Optional[dict[str, Any]]:
        return self._sent_messages[-1] if self._sent_messages else None


class MaiBotPlatformAdapter(PlatformAdapter):
    """真实 MaiBot 平台适配器。

    对接 MaiBot 的 PlatformIO (src/platform_io/) 和 SendService (src/services/)。

    发送接口:
        from services.send_service import send_session_message
        await send_session_message(message, ...)
    """

    def __init__(self, maibot_send_service=None):
        self._send_service = maibot_send_service
        self._available = maibot_send_service is not None
        self._init_tried = False

    def _ensure_init(self):
        """延迟初始化。"""
        if self._init_tried or self._available:
            return
        self._init_tried = True
        try:
            # MaiBot 运行时通过全局管理器获取发送服务
            from manager import GlobalManager
            gm = GlobalManager.get_instance()
            self._send_service = getattr(gm, "send_service", None)
            self._available = self._send_service is not None
        except Exception:
            pass

    def send_message(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
    ) -> bool:
        """同步发送（包装异步）。"""
        import asyncio
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.send_message_async(chat_id, content, reply_to))
        else:
            return asyncio.run_coroutine_threadsafe(
                self.send_message_async(chat_id, content, reply_to), loop
            ).result()

    async def send_message_async(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
    ) -> bool:
        """异步发送消息。"""
        self._ensure_init()
        if not self._available:
            # 降级到 Mock
            return MockPlatformAdapter().send_message(chat_id, content, reply_to)

        try:
            from services.send_service import send_session_message
            from common.models import SessionMessage, MaiMessage

            # 构造消息对象
            message = SessionMessage(
                chat_id=chat_id,
                content=content,
            )

            success = await send_session_message(
                message,
                set_reply=bool(reply_to),
                reply_message_id=reply_to,
                show_log=True,
            )
            return success
        except Exception as e:
            import logging
            logging.getLogger("pygal").error(f"消息发送失败: {e}")
            return False
