"""LPMM 记忆适配器 — 封装 MaiBot A-Memorix 的检索/写入接口。

MaiBot 的记忆系统位于 `src/A_memorix/`，核心是 SDKMemoryKernel：
  - search: 语义向量检索 / 时间检索 / 混合检索
  - write:  写入记忆节点（带 embedding）
  - graph:  知识图谱管理

本适配器将其封装为 Agent 可调用的简单接口。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional


class LPMMAdapter(ABC):
    """LPMM 记忆系统抽象接口。"""

    @abstractmethod
    def search(
        self,
        query: str,
        limit: int = 5,
        chat_id: str = "",
        person_id: str = "",
    ) -> list[dict[str, Any]]:
        """检索相关记忆。

        Args:
            query: 查询文本
            limit: 返回条数
            chat_id: 会话 ID（用于过滤）
            person_id: 人物 ID

        Returns:
            记忆条目列表，每条包含 content / timestamp / metadata 等
        """
        ...

    @abstractmethod
    def write(
        self,
        content: str,
        chat_id: str = "",
        person_id: str = "",
        metadata: Optional[dict] = None,
    ) -> bool:
        """写入一条记忆。

        Args:
            content: 记忆内容
            chat_id: 会话 ID
            person_id: 人物 ID
            metadata: 额外元数据

        Returns:
            是否写入成功
        """
        ...

    @abstractmethod
    def update_impression(
        self,
        person_id: str,
        impression_updates: dict[str, Any],
    ) -> bool:
        """更新对某人的印象标签。

        Args:
            person_id: 人物 ID
            impression_updates: 印象更新（如 {"kindness": +1}）

        Returns:
            是否更新成功
        """
        ...


class MockLPMMAdapter(LPMMAdapter):
    """Mock 记忆适配器 — 用于独立测试。"""

    def __init__(self):
        self._memories: list[dict[str, Any]] = []
        self._impressions: dict[str, dict[str, Any]] = {}

    def search(
        self,
        query: str,
        limit: int = 5,
        chat_id: str = "",
        person_id: str = "",
    ) -> list[dict[str, Any]]:
        """简单关键词匹配。"""
        results = []
        for mem in self._memories:
            if query.lower() in mem.get("content", "").lower():
                results.append(mem)
                if len(results) >= limit:
                    break
        return results

    def write(
        self,
        content: str,
        chat_id: str = "",
        person_id: str = "",
        metadata: Optional[dict] = None,
    ) -> bool:
        import time
        self._memories.append({
            "content": content,
            "chat_id": chat_id,
            "person_id": person_id,
            "metadata": metadata or {},
            "timestamp": time.time(),
        })
        return True

    def update_impression(
        self,
        person_id: str,
        impression_updates: dict[str, Any],
    ) -> bool:
        if person_id not in self._impressions:
            self._impressions[person_id] = {}
        self._impressions[person_id].update(impression_updates)
        return True

    @property
    def memory_count(self) -> int:
        return len(self._memories)


class MaiBotLPMMAdapter(LPMMAdapter):
    """真实 MaiBot A-Memorix 适配器。

    对接 MaiBot 的 SDKMemoryKernel (src/A_memorix/core/runtime/sdk_memory_kernel.py)。

    检索接口:
        kernel.search_memory(KernelSearchRequest(query=..., limit=..., mode="search"))
    写入接口:
        kernel.write_paragraph_vector_or_enqueue(...)
    """

    def __init__(self, maibot_memory_kernel=None):
        self._kernel = maibot_memory_kernel
        self._available = maibot_memory_kernel is not None
        self._init_tried = False

    def _ensure_init(self):
        """延迟初始化 SDKMemoryKernel。"""
        if self._init_tried or self._available:
            return
        self._init_tried = True
        try:
            from A_memorix.core.runtime.sdk_memory_kernel import SDKMemoryKernel
            # MaiBot 运行时会有全局 kernel 实例
            # 尝试从全局管理器获取
            from manager import GlobalManager
            gm = GlobalManager.get_instance()
            self._kernel = gm.memory_kernel
            self._available = self._kernel is not None
        except Exception:
            # 降级
            pass

    def search(
        self,
        query: str,
        limit: int = 5,
        chat_id: str = "",
        person_id: str = "",
    ) -> list[dict[str, Any]]:
        """检索记忆（同步包装异步）。"""
        import asyncio
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self._search_async(query, limit, chat_id, person_id))
        else:
            return asyncio.run_coroutine_threadsafe(
                self._search_async(query, limit, chat_id, person_id), loop
            ).result()

    async def _search_async(
        self, query: str, limit: int, chat_id: str, person_id: str
    ) -> list[dict[str, Any]]:
        self._ensure_init()
        if not self._available:
            return MockLPMMAdapter().search(query, limit, chat_id, person_id)

        try:
            from A_memorix.core.runtime.sdk_memory_kernel import KernelSearchRequest

            request = KernelSearchRequest(
                query=query,
                limit=limit,
                mode="search",
                chat_id=chat_id,
                person_id=person_id,
            )
            result = await self._kernel.search_memory(request)

            # 转换结果格式
            memories = []
            if isinstance(result, dict):
                items = result.get("results", result.get("items", []))
            elif isinstance(result, list):
                items = result
            else:
                items = []

            for item in items:
                if isinstance(item, dict):
                    memories.append({
                        "content": item.get("content", ""),
                        "timestamp": item.get("timestamp", ""),
                        "metadata": item.get("metadata", {}),
                    })
            return memories
        except Exception as e:
            import logging
            logging.getLogger("pygal").error(f"LPMM 检索失败: {e}")
            return []

    def write(
        self,
        content: str,
        chat_id: str = "",
        person_id: str = "",
        metadata: Optional[dict] = None,
    ) -> bool:
        """写入记忆（同步包装异步）。"""
        import asyncio
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self._write_async(content, chat_id, person_id, metadata))
        else:
            return asyncio.run_coroutine_threadsafe(
                self._write_async(content, chat_id, person_id, metadata), loop
            ).result()

    async def _write_async(
        self, content: str, chat_id: str, person_id: str, metadata: Optional[dict]
    ) -> bool:
        self._ensure_init()
        if not self._available:
            return MockLPMMAdapter().write(content, chat_id, person_id, metadata)

        try:
            # MaiBot A-Memorix 写入接口
            await self._kernel.write_paragraph_vector_or_enqueue(
                content=content,
                chat_id=chat_id,
                metadata=metadata or {},
            )
            return True
        except Exception as e:
            import logging
            logging.getLogger("pygal").error(f"LPMM 写入失败: {e}")
            return False

    def update_impression(
        self,
        person_id: str,
        impression_updates: dict[str, Any],
    ) -> bool:
        """更新印象（MaiBot person_info 模块）。"""
        self._ensure_init()
        if not self._available:
            return MockLPMMAdapter().update_impression(person_id, impression_updates)

        try:
            # MaiBot 的 person_info 模块管理人物画像
            # TODO: 对接 person_info 的印象更新接口
            # 当前先记录日志
            import logging
            logging.getLogger("pygal").info(
                f"印象更新: person={person_id}, updates={impression_updates}"
            )
            return True
        except Exception as e:
            import logging
            logging.getLogger("pygal").error(f"印象更新失败: {e}")
            return False
