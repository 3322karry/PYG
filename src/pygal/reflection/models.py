"""反思数据模型 — 记忆节点与印象更新。

反思引擎的输出产物：
  - MemoryNode:       结构化记忆节点，写入 LPMM
  - ImpressionUpdate: 对用户的印象标签更新，写入 LPMM 人物画像
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import time


@dataclass
class MemoryNode:
    """结构化记忆节点 — 反思提炼后的记忆单元。

    写入 LPMM 时包含:
      - content:    记忆内容（自然语言）
      - memory_type: 记忆类型（fact / preference / event / emotion）
      - importance:  重要性 0~1
      - source:      来源（reflection / conversation / observation）
      - related_persons: 相关人物 ID 列表
      - timestamp:   时间戳
    """
    content: str
    memory_type: str = "fact"  # fact / preference / event / emotion
    importance: float = 0.5
    source: str = "reflection"
    related_persons: list[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)

    def to_lpmm_metadata(self) -> dict:
        """转换为 LPMM 写入时的 metadata。"""
        return {
            "memory_type": self.memory_type,
            "importance": self.importance,
            "source": self.source,
            "related_persons": self.related_persons,
        }

    def to_dict(self) -> dict:
        return {
            "content": self.content,
            "memory_type": self.memory_type,
            "importance": self.importance,
            "source": self.source,
            "related_persons": list(self.related_persons),
            "timestamp": self.timestamp,
        }


@dataclass
class ImpressionUpdate:
    """对某用户的印象更新。

    印象维度（可扩展）:
      - kindness:     善良度
      - intelligence:  智力度
      - humor:         幽默度
      - openness:      开放度
      - trust:         信任度
    """
    person_id: str
    person_name: str = ""
    updates: dict[str, int] = field(default_factory=dict)  # 维度 → 增量
    summary: str = ""  # 自然语言印象摘要
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "person_id": self.person_id,
            "person_name": self.person_name,
            "updates": dict(self.updates),
            "summary": self.summary,
            "timestamp": self.timestamp,
        }
