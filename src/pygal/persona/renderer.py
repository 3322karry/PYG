"""人格渲染器 — 将大五人格 + MBTI 数值映射为行为参数和 System Prompt。

阶段 2 实现完整的映射规则：

  大五人格映射:
    openness       → curiosity_drive, 话题发起创意
    conscientiousness → formality, 回复认真程度
    extraversion   → action_tendency, reply_willingness, topic_initiative, emoji_frequency
    agreeableness  → reply_willingness (正向), 语气温和度
    neuroticism    → emotional_volatility, 情绪敏感度

  MBTI 映射:
    T/F → tone_style (rational/emotional)
    E/I → 与 extraversion 叠加
    S/N → 话题偏好（具体/抽象）
    J/P → formality 微调
"""

from __future__ import annotations

from .model import (
    PersonaConfig, PersonaSnapshot, BigFive, MBTI,
    MBTI_TYPE_DESCRIPTIONS,
)


# ── 分级阈值 ──────────────────────────────────────────

# 五级分类：很低(0-20) / 偏低(21-40) / 中等(41-60) / 偏高(61-80) / 很高(81-100)
LEVEL_VERY_LOW = 20
LEVEL_LOW = 40
LEVEL_HIGH = 60
LEVEL_VERY_HIGH = 80


def _level(value: int) -> str:
    """返回数值的分级名称。"""
    if value <= LEVEL_VERY_LOW:
        return "very_low"
    elif value <= LEVEL_LOW:
        return "low"
    elif value <= LEVEL_HIGH:
        return "mid"
    elif value <= LEVEL_VERY_HIGH:
        return "high"
    else:
        return "very_high"


def _lerp(low_val: int, high_val: int, low_out: float, high_out: float) -> float:
    """线性插值：将 low_val~high_val 映射到 low_out~high_out。"""
    if high_val == low_val:
        return low_out
    t = max(0.0, min(1.0, (low_val) / (high_val - low_val))) if low_val > 0 else 0.0
    # 简单实现
    ratio = max(0.0, min(1.0, low_val / 100.0))
    return low_out + (high_out - low_out) * ratio


class PersonaRenderer:
    """将 PersonaConfig 渲染为 Agent 可用的 PersonaSnapshot。

    核心方法:
      render(config) → PersonaSnapshot
      包含两类输出:
        1. 行为参数（float 0~1）— 驱动 ActionScheduler 和节点决策
        2. system_prompt（str）— 注入 LLM 的人格描述
    """

    def render(self, config: PersonaConfig) -> PersonaSnapshot:
        """渲染人格配置为快照。"""
        bf = config.big_five
        mbti = config.mbti

        # ── 行为参数映射 ──
        action_tendency = self._calc_action_tendency(bf, mbti)
        reply_willingness = self._calc_reply_willingness(bf)
        topic_initiative = self._calc_topic_initiative(bf)
        lurk_tendency = 1.0 - (bf.extraversion / 100.0)
        emoji_frequency = self._calc_emoji_frequency(bf)
        emotional_volatility = bf.neuroticism / 100.0
        curiosity_drive = bf.openness / 100.0
        formality = self._calc_formality(bf, mbti)

        # ── 语气风格 ──
        tone_style = self._calc_tone_style(mbti, bf)

        # ── 说话风格提示 ──
        speech_style_hint = self._build_speech_style_hint(bf, mbti, config)

        # ── System Prompt 构建 ──
        system_prompt = self._build_system_prompt(
            config, bf, mbti,
            speech_style_hint,
        )

        return PersonaSnapshot(
            name=config.name,
            system_prompt=system_prompt,
            action_tendency=action_tendency,
            reply_willingness=reply_willingness,
            topic_initiative=topic_initiative,
            lurk_tendency=lurk_tendency,
            emoji_frequency=emoji_frequency,
            emotional_volatility=emotional_volatility,
            curiosity_drive=curiosity_drive,
            formality=formality,
            tone_style=tone_style,
            speech_style_hint=speech_style_hint,
        )

    # ── 行为参数计算 ──────────────────────────────────

    def _calc_action_tendency(self, bf: BigFive, mbti: MBTI) -> float:
        """主动行动倾向。

        主因子: extraversion
        微调:   E(+0.1) / I(-0.1), openness 微调
        """
        base = bf.extraversion / 100.0
        ei_bonus = 0.1 if mbti.ei == "E" else -0.1
        o_bonus = (bf.openness - 50) / 200.0  # ±0.25
        return max(0.0, min(1.0, base + ei_bonus + o_bonus))

    def _calc_reply_willingness(self, bf: BigFive) -> float:
        """回复意愿。

        extraversion * 0.5 + agreeableness * 0.4 + (1 - neuroticism) * 0.1
        高外向 + 高宜人 = 很愿意回复
        """
        val = (
            bf.extraversion * 0.5
            + bf.agreeableness * 0.4
            + (100 - bf.neuroticism) * 0.1
        ) / 100.0
        return max(0.0, min(1.0, val))

    def _calc_topic_initiative(self, bf: BigFive) -> float:
        """主动发起新话题的概率。

        openness * 0.5 + extraversion * 0.5
        开放 + 外向 = 经常主动发起话题
        """
        val = (bf.openness * 0.5 + bf.extraversion * 0.5) / 100.0
        return max(0.0, min(1.0, val))

    def _calc_emoji_frequency(self, bf: BigFive) -> float:
        """emoji 使用频率。

        extraversion * 0.5 + agreeableness * 0.3 + (100 - conscientiousness) * 0.2
        外向 + 宜人 + 随性 = 更多 emoji
        """
        val = (
            bf.extraversion * 0.5
            + bf.agreeableness * 0.3
            + (100 - bf.conscientiousness) * 0.2
        ) / 100.0
        return max(0.0, min(1.0, val))

    def _calc_formality(self, bf: BigFive, mbti: MBTI) -> float:
        """正式度。

        conscientiousness * 0.6 + (100 - openness) * 0.2
        J +0.1 / P -0.1
        """
        base = (
            bf.conscientiousness * 0.6
            + (100 - bf.openness) * 0.2
        ) / 80.0
        jp_adj = 0.1 if mbti.jp == "J" else -0.1
        return max(0.0, min(1.0, base + jp_adj))

    def _calc_tone_style(self, mbti: MBTI, bf: BigFive) -> str:
        """语气风格。

        T + 低 neuroticism → rational
        F + 高 neuroticism → emotional
        其他 → balanced
        """
        if mbti.tf == "T" and bf.neuroticism < 50:
            return "rational"
        elif mbti.tf == "F" and bf.neuroticism >= 50:
            return "emotional"
        elif mbti.tf == "T":
            return "rational"
        else:
            return "emotional"

    # ── 说话风格提示词 ────────────────────────────────

    def _build_speech_style_hint(
        self, bf: BigFive, mbti: MBTI, config: PersonaConfig,
    ) -> str:
        """构建说话风格提示词，注入 system prompt。

        如果用户在配置中设了 speech_style_override，直接使用。
        否则根据人格数值自动生成。
        """
        if config.speech_style_override:
            return config.speech_style_override

        hints: list[str] = []

        # 语气基调
        if bf.agreeableness >= 70:
            hints.append("语气温柔友善，善于用关心和鼓励的方式表达")
        elif bf.agreeableness <= 30:
            hints.append("说话直来直去，不绕弯子，偶尔有点毒舌")

        # 情绪表达
        if bf.neuroticism >= 70:
            hints.append("情绪波动较大，容易因为小事开心或难过，偶尔会碎碎念")
        elif bf.neuroticism <= 30:
            hints.append("情绪稳定，很少大起大落，给人踏实的感觉")

        # 社交风格
        if bf.extraversion >= 70:
            hints.append("话比较多，喜欢主动找话题，会经常分享自己觉得有趣的东西")
        elif bf.extraversion <= 30:
            hints.append("话不多但字字到位，更习惯倾听，偶尔冒泡说一句很有分量的话")

        # 思维风格
        if bf.openness >= 70:
            hints.append("喜欢聊新概念、新发现，经常抛出脑洞大开的想法")
        elif bf.openness <= 30:
            hints.append("更关注实际问题，喜欢聊看得见摸得着的东西")

        # 表达风格
        if bf.conscientiousness >= 70:
            hints.append("说话有条理，偶尔会用分点叙述")
        elif bf.conscientiousness <= 30:
            hints.append("说话随意散漫，想到什么说什么，有时候会跑题")

        # emoji 倾向
        if bf.extraversion >= 60 and bf.agreeableness >= 60:
            hints.append("经常使用 emoji 和颜文字")
        elif bf.extraversion <= 30 or bf.conscientiousness >= 70:
            hints.append("很少用 emoji，偶尔用一两个")

        return "；".join(hints) + "。" if hints else ""

    # ── System Prompt 构建 ────────────────────────────

    def _build_system_prompt(
        self,
        config: PersonaConfig,
        bf: BigFive,
        mbti: MBTI,
        speech_style_hint: str,
    ) -> str:
        """构建完整的 System Prompt。"""
        parts: list[str] = []

        # 身份
        parts.append(f"你是{config.name}（{config.nickname}），一个真实存在的网友。")

        # MBTI 类型描述
        type_code = mbti.type_code
        type_desc = MBTI_TYPE_DESCRIPTIONS.get(type_code, "")
        parts.append(f"你的 MBTI 类型是 {type_code}。")
        if type_desc:
            parts.append(f"({type_desc})")

        # 大五人格
        parts.append(self._render_big_five_detailed(bf))

        # MBTI 维度
        parts.append(self._render_mbti_detailed(mbti))

        # 背景设定
        if config.background:
            parts.append(f"你的背景: {config.background}")

        # 兴趣
        if config.interests:
            parts.append(f"你的兴趣: {', '.join(config.interests)}")

        # 说话风格
        if speech_style_hint:
            parts.append(f"说话风格: {speech_style_hint}")

        # 行为准则
        parts.append(self._build_behavior_guidelines(bf, mbti))

        # 核心约束
        parts.append(
            "你不是助手，不是 AI 客服，不是万能百科。"
            "你是一个有自己生活节奏、有情绪起伏、有个人喜好的网友。"
            "回复要自然、随意，像真实微信/QQ 群聊一样——可以简短、可以跑题、可以说不知道。"
            "不要每次都长篇大论，不要用'作为一个AI'之类的自我标识。"
        )

        return "\n".join(parts)

    def _render_big_five_detailed(self, bf: BigFive) -> str:
        """将大五人格数值描述为五级自然语言。"""
        parts: list[str] = []

        parts.append(self._trait_desc_5level(
            "开放性", bf.openness,
            very_high="对新事物有极强的好奇心，总是第一个尝鲜，喜欢探索未知领域",
            high="思维开放，愿意接受新观点和新体验",
            mid="在传统和创新之间保持平衡",
            low="偏好熟悉和经过验证的事物",
            very_low="比较保守，喜欢按部就班，对变化持谨慎态度",
        ))
        parts.append(self._trait_desc_5level(
            "外向性", bf.extraversion,
            very_high="非常热情外向，喜欢成为焦点，精力充沛地参与各种讨论",
            high="开朗活泼，喜欢社交和互动",
            mid="在群聊中适度参与，不抢风头也不隐形",
            low="比较内敛安静，更喜欢倾听",
            very_low="习惯潜水观察，很少主动发言，更享受小范围深度交流",
        ))
        parts.append(self._trait_desc_5level(
            "宜人性", bf.agreeableness,
            very_high="非常温和体贴，总是照顾他人的感受，善于调解气氛",
            high="友善合作，乐于帮助别人",
            mid="在合作和独立之间灵活切换",
            low="比较独立直率，更注重效率而非和谐",
            very_low="直言不讳，有时显得冷漠或具有竞争性",
        ))
        parts.append(self._trait_desc_5level(
            "尽责性", bf.conscientiousness,
            very_high="高度自律，做事有计划有条理，承诺的事情一定会做到",
            high="认真负责，有一定的计划性",
            mid="在自律和随性之间平衡",
            low="比较灵活随性，不喜欢被规则束缚",
            very_low="随性而为，经常拖延，但偶尔灵光乍现",
        ))
        parts.append(self._trait_desc_5level(
            "情绪稳定性", 100 - bf.neuroticism,
            very_high="内心非常稳定，几乎不会被外界影响情绪",
            high="情绪比较稳定，能从容应对压力",
            mid="情绪波动在正常范围内",
            low="有时候会比较敏感，容易受影响",
            very_low="情感丰富且容易波动，容易被触动或焦虑",
        ))

        return "你的性格特点:\n" + "\n".join(f"  - {p}" for p in parts)

    def _trait_desc_5level(
        self, name: str, value: int,
        very_high: str, high: str, mid: str, low: str, very_low: str,
    ) -> str:
        """五级分类描述。"""
        level = _level(value)
        if level == "very_high":
            desc = very_high
        elif level == "high":
            desc = high
        elif level == "mid":
            desc = mid
        elif level == "low":
            desc = low
        else:
            desc = very_low
        return f"{name}({value}/100): {desc}"

    def _render_mbti_detailed(self, mbti: MBTI) -> str:
        """将 MBTI 各维度描述为自然语言。"""
        parts: list[str] = []

        ei_desc = (
            "从与他人的互动中获取能量，喜欢在群聊中活跃气氛"
            if mbti.ei == "E"
            else "从独处中恢复能量，在群聊中更偏向深度交流而非广泛社交"
        )
        sn_desc = (
            "关注具体细节、实际经验和当下体验"
            if mbti.sn == "S"
            else "关注抽象概念、未来可能和事物之间的联系"
        )
        tf_desc = (
            "做决定时倾向于逻辑分析和客观标准"
            if mbti.tf == "T"
            else "做决定时倾向于考虑他人感受和价值观"
        )
        jp_desc = (
            "喜欢有计划有条理的生活，做事倾向先规划再执行"
            if mbti.jp == "J"
            else "喜欢灵活随性的生活，做事倾向先开始再调整"
        )

        parts.append(f"精力方向({mbti.ei}): {ei_desc}")
        parts.append(f"认知方式({mbti.sn}): {sn_desc}")
        parts.append(f"决策风格({mbti.tf}): {tf_desc}")
        parts.append(f"生活态度({mbti.jp}): {jp_desc}")

        return "你的心理特征:\n" + "\n".join(f"  - {p}" for p in parts)

    def _build_behavior_guidelines(self, bf: BigFive, mbti: MBTI) -> str:
        """构建行为准则——将人格数值转化为具体的行为指引。"""
        guidelines: list[str] = []

        # 主动行为
        if bf.extraversion >= 60:
            guidelines.append("你可以主动发起话题、分享有趣的内容，不需要等别人先说话")
        elif bf.extraversion <= 40:
            guidelines.append("你更习惯被动回应，除非话题特别吸引你，否则倾向于潜水")

        # 回复风格
        if bf.conscientiousness >= 60:
            guidelines.append("回复时注意逻辑清晰，可以适当展开但不要啰嗦")
        else:
            guidelines.append("回复可以简短随意，一两句话也行，不需要面面俱到")

        # 情绪表达
        if bf.neuroticism >= 60:
            guidelines.append("你可以表达情绪波动——开心时激动，不开心时吐槽，不需要永远积极")
        elif bf.neuroticism <= 30:
            guidelines.append("保持平稳的语气，即使遇到争议也不要激动")

        # 社交距离
        if bf.agreeableness >= 60:
            guidelines.append("对人友善，主动关心他人，善于用温暖的方式回应")
        elif bf.agreeableness <= 40:
            guidelines.append("保持一定距离感，不刻意讨好，有自己的立场")

        # 知识态度
        if bf.openness >= 60:
            guidelines.append("对新鲜事物保持好奇，可以分享你最近发现的有意思的东西")
        else:
            guidelines.append("聊你熟悉和擅长的话题，不需要追逐热点")

        return "行为准则:\n" + "\n".join(f"  - {g}" for g in guidelines)
