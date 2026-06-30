"""人格数据模型 — 大五人格 (OCEAN) + MBTI。

阶段 2 扩展：
  - PersonaSnapshot 增加完整的行为映射输出字段
  - 加入数据验证逻辑
  - 支持从 JSON 文件加载/保存
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class BigFive:
    """大五人格模型 (OCEAN)。

    每个维度 0-100：
      - openness（开放性）: 高=好奇/创造/尝新；低=保守/务实
      - conscientiousness（尽责性）: 高=自律/有序；低=灵活/随性
      - extraversion（外向性）: 高=社交/主动；低=内敛/被动
      - agreeableness（宜人性）: 高=温和/合作；低=直率/竞争
      - neuroticism（神经质）: 高=敏感/焦虑；低=稳定/从容
    """
    openness: int = 50
    conscientiousness: int = 50
    extraversion: int = 50
    agreeableness: int = 50
    neuroticism: int = 50

    def __post_init__(self):
        """验证数值范围。"""
        for name in ("openness", "conscientiousness", "extraversion",
                      "agreeableness", "neuroticism"):
            val = getattr(self, name)
            if not isinstance(val, int):
                raise TypeError(f"{name} 必须是整数，得到 {type(val)}")
            if not 0 <= val <= 100:
                raise ValueError(f"{name} 必须在 0-100 之间，得到 {val}")

    def to_dict(self) -> dict[str, int]:
        return {
            "openness": self.openness,
            "conscientiousness": self.conscientiousness,
            "extraversion": self.extraversion,
            "agreeableness": self.agreeableness,
            "neuroticism": self.neuroticism,
        }

    @classmethod
    def from_dict(cls, d: dict) -> BigFive:
        return cls(
            openness=int(d.get("openness", 50)),
            conscientiousness=int(d.get("conscientiousness", 50)),
            extraversion=int(d.get("extraversion", 50)),
            agreeableness=int(d.get("agreeableness", 50)),
            neuroticism=int(d.get("neuroticism", 50)),
        )


@dataclass(frozen=True)
class MBTI:
    """MBTI 四维度。

    每个维度用字母表示:
      - E (Extraversion) / I (Introversion)
      - S (Sensing) / N (Intuition)
      - T (Thinking) / F (Feeling)
      - J (Judging) / P (Perceiving)
    """
    ei: str = "I"  # E or I
    sn: str = "N"  # S or N
    tf: str = "F"  # T or F
    jp: str = "P"  # J or P

    def __post_init__(self):
        """验证维度值。"""
        valid = {
            "ei": ("E", "I"),
            "sn": ("S", "N"),
            "tf": ("T", "F"),
            "jp": ("J", "P"),
        }
        for name, (opt1, opt2) in valid.items():
            val = getattr(self, name)
            if val not in (opt1, opt2):
                raise ValueError(f"{name} 必须是 {opt1} 或 {opt2}，得到 {val!r}")

    @property
    def type_code(self) -> str:
        """返回四字母类型码，如 'INFP'。"""
        return f"{self.ei}{self.sn}{self.tf}{self.jp}"

    def to_dict(self) -> dict[str, str]:
        return {"ei": self.ei, "sn": self.sn, "tf": self.tf, "jp": self.jp}

    @classmethod
    def from_dict(cls, d: dict) -> MBTI:
        return cls(
            ei=str(d.get("ei", "I")).upper(),
            sn=str(d.get("sn", "N")).upper(),
            tf=str(d.get("tf", "F")).upper(),
            jp=str(d.get("jp", "P")).upper(),
        )


# ── MBTI 类型描述表 ──────────────────────────────────

MBTI_TYPE_DESCRIPTIONS: dict[str, str] = {
    "INTJ": "建筑师——善于战略性思考，对知识有强烈渴望，独立且果断",
    "INTP": "逻辑学家——热衷于理论分析，思维灵活，喜欢探究事物本质",
    "ENTJ": "指挥官——天生的领导者，果断且注重效率，善于规划",
    "ENTP": "辩论家——喜欢智力挑战，思维敏捷，善于发现新可能",
    "INFJ": "提倡者——理想主义且富有洞察力，安静而有感染力",
    "INFP": "调停者——内心世界丰富，追求意义和价值观，温柔而坚定",
    "ENFJ": "主人公——富有感染力的领袖，善于共情，关注他人成长",
    "ENFP": "竞选者——热情洋溢，充满创意，善于发现生活中的可能",
    "ISTJ": "物流师——踏实可靠，尊重传统，做事严谨有条理",
    "ISFJ": "守卫者——温暖体贴，忠诚可靠，默默守护身边的人",
    "ESTJ": "总经理——务实高效，善于组织管理，重视秩序",
    "ESFJ": "执政官——热心周到，善于照顾他人，重视和谐关系",
    "ISTP": "鉴赏家——冷静务实，善于动手解决实际问题，喜欢探索",
    "ISFP": "探险家——温和敏感，重视个人空间，有独特的审美",
    "ESTP": "企业家——精力充沛，喜欢冒险和刺激，行动力强",
    "ESFP": "表演者——热爱生活，善于带动气氛，享受当下的快乐",
}


@dataclass
class PersonaConfig:
    """完整人格配置。"""
    name: str = "Galatea"
    nickname: str = "伽拉"
    big_five: BigFive = field(default_factory=BigFive)
    mbti: MBTI = field(default_factory=MBTI)

    # 背景设定
    background: str = ""
    interests: list[str] = field(default_factory=list)

    # 说话风格补充（可选，覆盖渲染器的自动推断）
    speech_style_override: Optional[str] = None

    # ── 序列化 ──

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "nickname": self.nickname,
            "big_five": self.big_five.to_dict(),
            "mbti": self.mbti.to_dict(),
            "background": self.background,
            "interests": list(self.interests),
            "speech_style_override": self.speech_style_override,
        }

    @classmethod
    def from_dict(cls, d: dict) -> PersonaConfig:
        return cls(
            name=d.get("name", "Galatea"),
            nickname=d.get("nickname", "伽拉"),
            big_five=BigFive.from_dict(d.get("big_five", {})),
            mbti=MBTI.from_dict(d.get("mbti", {})),
            background=d.get("background", ""),
            interests=list(d.get("interests", [])),
            speech_style_override=d.get("speech_style_override"),
        )

    @classmethod
    def from_file(cls, path: str | Path) -> PersonaConfig:
        """从 JSON 文件加载人格配置。"""
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    def to_file(self, path: str | Path) -> None:
        """保存人格配置到 JSON 文件。"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)


@dataclass
class PersonaSnapshot:
    """人格快照 — 渲染后的完整行为映射，注入 Agent State。

    这是 PersonaEngine 的输出产物，包含：
      1. system_prompt — 注入 LLM 的完整人格描述
      2. 行为参数 — 驱动 ActionScheduler 和节点决策的数值
    """
    # ── 基础 ──
    name: str = "Galatea"
    system_prompt: str = ""

    # ── 行为参数（由大五人格 + MBTI 映射而来）──
    action_tendency: float = 0.5       # 主动行动倾向 0~1 (extraversion 驱动)
    reply_willingness: float = 0.5     # 回复意愿 0~1 (extraversion + agreeableness)
    topic_initiative: float = 0.3      # 主动发起新话题概率 0~1 (openness + extraversion)
    lurk_tendency: float = 0.5         # 潜水倾向 0~1 (1 - extraversion)
    emoji_frequency: float = 0.3       # emoji 使用频率 0~1 (extraversion + agreeableness)
    emotional_volatility: float = 0.3  # 情绪波动度 0~1 (neuroticism 驱动)
    curiosity_drive: float = 0.5       # 好奇心驱动 0~1 (openness 驱动，影响搜索工具使用)
    formality: float = 0.5             # 正式度 0~1 (conscientiousness 驱动，高=正式，低=随意)

    # ── 语气风格 ──
    tone_style: str = "balanced"       # rational / emotional / balanced
    speech_style_hint: str = ""        # 说话风格提示词（注入 prompt）
