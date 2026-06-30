"""pyGal 阶段 2 测试 — 人格配置与行为映射。"""
from __future__ import annotations

import json
import pytest
from pathlib import Path

from pygal.persona.model import (
    PersonaConfig, BigFive, MBTI, PersonaSnapshot,
    MBTI_TYPE_DESCRIPTIONS,
)
from pygal.persona.renderer import PersonaRenderer, _level

CONFIG_DIR = Path(__file__).parent.parent / "config"


# ── 辅助函数 ──

@pytest.fixture
def renderer() -> PersonaRenderer:
    return PersonaRenderer()


@pytest.fixture
def default_config() -> PersonaConfig:
    return PersonaConfig()


@pytest.fixture
def extrovert_config() -> PersonaConfig:
    return PersonaConfig(
        name="小鹿",
        nickname="鹿鹿",
        big_five=BigFive(
            openness=85, conscientiousness=30,
            extraversion=90, agreeableness=75, neuroticism=70,
        ),
        mbti=MBTI(ei="E", sn="N", tf="F", jp="P"),
        background="大学生，气氛组担当",
        interests=["追星", "奶茶"],
    )


@pytest.fixture
def introvert_config() -> PersonaConfig:
    return PersonaConfig(
        name="冷锋",
        nickname="锋哥",
        big_five=BigFive(
            openness=45, conscientiousness=80,
            extraversion=25, agreeableness=30, neuroticism=35,
        ),
        mbti=MBTI(ei="I", sn="S", tf="T", jp="J"),
        background="程序员，话少但靠谱",
        interests=["系统架构", "象棋"],
    )


# ── 1. 数据模型验证 ──

class TestDataModel:

    def test_big_five_defaults(self):
        bf = BigFive()
        assert bf.openness == 50
        assert all(50 == getattr(bf, f) for f in
                   ("openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"))

    def test_big_five_validation_high(self):
        with pytest.raises(ValueError):
            BigFive(openness=101)

    def test_big_five_validation_low(self):
        with pytest.raises(ValueError):
            BigFive(extraversion=-1)

    def test_big_five_must_be_int(self):
        with pytest.raises(TypeError):
            BigFive(openness=50.5)

    def test_mbti_type_code(self):
        mbti = MBTI(ei="I", sn="N", tf="T", jp="J")
        assert mbti.type_code == "INTJ"

    def test_mbti_validation(self):
        with pytest.raises(ValueError):
            MBTI(ei="X")

    def test_mbti_case_insensitive(self):
        mbti = MBTI.from_dict({"ei": "e", "sn": "n", "tf": "f", "jp": "p"})
        assert mbti.type_code == "ENFP"

    def test_persona_config_roundtrip(self, extrovert_config):
        d = extrovert_config.to_dict()
        restored = PersonaConfig.from_dict(d)
        assert restored.name == extrovert_config.name
        assert restored.big_five.extraversion == extrovert_config.big_five.extraversion
        assert restored.mbti.type_code == extrovert_config.mbti.type_code

    def test_persona_config_from_file(self):
        path = CONFIG_DIR / "persona.json"
        config = PersonaConfig.from_file(path)
        assert config.name == "Galatea"
        assert config.big_five.openness == 75

    def test_persona_config_to_file(self, tmp_path, extrovert_config):
        path = tmp_path / "test_persona.json"
        extrovert_config.to_file(path)
        loaded = PersonaConfig.from_file(path)
        assert loaded.name == "小鹿"
        assert loaded.big_five.extraversion == 90


# ── 2. MBTI 类型描述表 ──

class TestMBTIDescriptions:

    def test_all_16_types_have_descriptions(self):
        """16 种 MBTI 类型都应该有描述。"""
        types = [
            f"{ei}{sn}{tf}{jp}"
            for ei in "EI" for sn in "SN" for tf in "TF" for jp in "JP"
        ]
        for t in types:
            assert t in MBTI_TYPE_DESCRIPTIONS, f"缺少 {t} 的描述"

    def test_description_is_meaningful(self):
        for code, desc in MBTI_TYPE_DESCRIPTIONS.items():
            assert len(desc) > 10, f"{code} 描述太短"


# ── 3. 行为参数映射 ──

class TestBehaviorMapping:

    def test_action_tendency_extrovert_high(self, renderer, extrovert_config):
        """外向型人格的主动行动倾向应该高。"""
        snapshot = renderer.render(extrovert_config)
        assert snapshot.action_tendency > 0.7

    def test_action_tendency_introvert_low(self, renderer, introvert_config):
        """内向型人格的主动行动倾向应该低。"""
        snapshot = renderer.render(introvert_config)
        assert snapshot.action_tendency < 0.4

    def test_reply_willingness_extrovert(self, renderer, extrovert_config):
        """外向 + 宜人 = 回复意愿高。"""
        snapshot = renderer.render(extrovert_config)
        assert snapshot.reply_willingness > 0.7

    def test_reply_willingness_introvert(self, renderer, introvert_config):
        """内向 + 低宜人 = 回复意愿低。"""
        snapshot = renderer.render(introvert_config)
        assert snapshot.reply_willingness < 0.5

    def test_topic_initiative_open_extrovert(self, renderer, extrovert_config):
        """高开放 + 高外向 = 话题发起概率高。"""
        snapshot = renderer.render(extrovert_config)
        assert snapshot.topic_initiative > 0.7

    def test_topic_initiative_closed_introvert(self, renderer, introvert_config):
        """低开放 + 低外向 = 话题发起概率低。"""
        snapshot = renderer.render(introvert_config)
        assert snapshot.topic_initiative < 0.4

    def test_lurk_tendency_inverse_extraversion(self, renderer):
        """潜水倾向应与外向性反相关。"""
        extro = renderer.render(PersonaConfig(big_five=BigFive(extraversion=90)))
        intro = renderer.render(PersonaConfig(big_five=BigFive(extraversion=10)))
        assert extro.lurk_tendency < intro.lurk_tendency
        assert extro.lurk_tendency < 0.2
        assert intro.lurk_tendency > 0.8

    def test_emoji_frequency_high_extrovert(self, renderer, extrovert_config):
        """高外向 + 高宜人 + 低尽责 = emoji 多。"""
        snapshot = renderer.render(extrovert_config)
        assert snapshot.emoji_frequency > 0.6

    def test_emoji_frequency_low_introvert(self, renderer, introvert_config):
        """低外向 + 低宜人 = emoji 少。"""
        snapshot = renderer.render(introvert_config)
        assert snapshot.emoji_frequency < 0.4

    def test_emotional_volatility_neuroticism(self, renderer):
        """情绪波动度应与神经质正相关。"""
        stable = renderer.render(PersonaConfig(big_five=BigFive(neuroticism=10)))
        volatile = renderer.render(PersonaConfig(big_five=BigFive(neuroticism=90)))
        assert stable.emotional_volatility < volatile.emotional_volatility
        assert stable.emotional_volatility < 0.15
        assert volatile.emotional_volatility > 0.85

    def test_curiosity_drive_openness(self, renderer):
        """好奇心驱动应与开放性正相关。"""
        low_o = renderer.render(PersonaConfig(big_five=BigFive(openness=20)))
        high_o = renderer.render(PersonaConfig(big_five=BigFive(openness=90)))
        assert low_o.curiosity_drive < high_o.curiosity_drive

    def test_formality_conscientiousness(self, renderer, introvert_config):
        """高尽责 + J = 正式度高。"""
        snapshot = renderer.render(introvert_config)
        assert snapshot.formality > 0.6

    def test_formality_low(self, renderer, extrovert_config):
        """低尽责 + P = 正式度低。"""
        snapshot = renderer.render(extrovert_config)
        assert snapshot.formality < 0.5

    def test_all_params_in_range(self, renderer, extrovert_config, introvert_config):
        """所有行为参数应该在 0~1 范围内。"""
        for config in [extrovert_config, introvert_config, PersonaConfig()]:
            s = renderer.render(config)
            for attr in ("action_tendency", "reply_willingness", "topic_initiative",
                         "lurk_tendency", "emoji_frequency", "emotional_volatility",
                         "curiosity_drive", "formality"):
                val = getattr(s, attr)
                assert 0.0 <= val <= 1.0, f"{attr}={val} 超出范围"


# ── 4. 语气风格映射 ──

class TestToneStyle:

    def test_thinking_type_rational(self, renderer):
        """T 型 + 低 neuroticism = rational。"""
        config = PersonaConfig(
            big_five=BigFive(neuroticism=30),
            mbti=MBTI(tf="T"),
        )
        snapshot = renderer.render(config)
        assert snapshot.tone_style == "rational"

    def test_feeling_type_emotional(self, renderer):
        """F 型 + 高 neuroticism = emotional。"""
        config = PersonaConfig(
            big_five=BigFive(neuroticism=70),
            mbti=MBTI(tf="F"),
        )
        snapshot = renderer.render(config)
        assert snapshot.tone_style == "emotional"


# ── 5. System Prompt 渲染 ──

class TestSystemPrompt:

    def test_prompt_contains_name(self, renderer, extrovert_config):
        snapshot = renderer.render(extrovert_config)
        assert "小鹿" in snapshot.system_prompt

    def test_prompt_contains_mbti_code(self, renderer, introvert_config):
        snapshot = renderer.render(introvert_config)
        assert "ISTJ" in snapshot.system_prompt

    def test_prompt_contains_mbti_description(self, renderer, extrovert_config):
        snapshot = renderer.render(extrovert_config)
        assert "竞选者" in snapshot.system_prompt

    def test_prompt_contains_big_five_scores(self, renderer):
        config = PersonaConfig(big_five=BigFive(extraversion=77))
        snapshot = renderer.render(config)
        assert "77" in snapshot.system_prompt

    def test_prompt_contains_background(self, renderer, extrovert_config):
        snapshot = renderer.render(extrovert_config)
        assert "大学生" in snapshot.system_prompt

    def test_prompt_contains_interests(self, renderer, extrovert_config):
        snapshot = renderer.render(extrovert_config)
        assert "追星" in snapshot.system_prompt

    def test_prompt_contains_behavior_guidelines(self, renderer, extrovert_config):
        snapshot = renderer.render(extrovert_config)
        assert "行为准则" in snapshot.system_prompt

    def test_prompt_not_ai_assistant(self, renderer, extrovert_config):
        snapshot = renderer.render(extrovert_config)
        assert "不是助手" in snapshot.system_prompt
        assert "不是 AI 客服" in snapshot.system_prompt

    def test_speech_style_override(self, renderer):
        """speech_style_override 应覆盖自动生成的说话风格。"""
        config = PersonaConfig(
            name="测试",
            speech_style_override="说话极简，偶尔冷幽默。",
        )
        snapshot = renderer.render(config)
        assert "说话极简" in snapshot.system_prompt

    def test_speech_style_auto_generated(self, renderer, extrovert_config):
        """没有 override 时应自动生成说话风格。"""
        snapshot = renderer.render(extrovert_config)
        assert snapshot.speech_style_hint
        assert len(snapshot.speech_style_hint) > 10


# ── 6. 分级函数 ──

class TestLevelFunction:

    def test_very_low(self):
        assert _level(0) == "very_low"
        assert _level(20) == "very_low"

    def test_low(self):
        assert _level(21) == "low"
        assert _level(40) == "low"

    def test_mid(self):
        assert _level(41) == "mid"
        assert _level(60) == "mid"

    def test_high(self):
        assert _level(61) == "high"
        assert _level(80) == "high"

    def test_very_high(self):
        assert _level(81) == "very_high"
        assert _level(100) == "very_high"


# ── 7. 预设角色对比 ──

class TestPresetPersonas:

    def test_extrovert_vs_introvert_contrast(self, renderer, extrovert_config, introvert_config):
        """外向型和内向型人格应该产生明显不同的行为参数。"""
        extro = renderer.render(extrovert_config)
        intro = renderer.render(introvert_config)

        # 外向型应该全面更主动
        assert extro.action_tendency > intro.action_tendency
        assert extro.reply_willingness > intro.reply_willingness
        assert extro.topic_initiative > intro.topic_initiative
        assert extro.lurk_tendency < intro.lurk_tendency
        assert extro.emoji_frequency > intro.emoji_frequency

    def test_introvert_more_formal(self, renderer, extrovert_config, introvert_config):
        """内向型（高尽责 + J）应该更正式。"""
        extro = renderer.render(extrovert_config)
        intro = renderer.render(introvert_config)
        assert intro.formality > extro.formality

    def test_extrovert_more_emotional(self, renderer, extrovert_config, introvert_config):
        """外向型（高 neuroticism + F）应该情绪波动更大。"""
        extro = renderer.render(extrovert_config)
        intro = renderer.render(introvert_config)
        assert extro.emotional_volatility > intro.emotional_volatility

    def test_prompt_length_reasonable(self, renderer, extrovert_config):
        """System Prompt 不应太短。"""
        snapshot = renderer.render(extrovert_config)
        assert len(snapshot.system_prompt) > 500

    def test_config_file_loads_correctly(self):
        """从文件加载的配置应正确解析。"""
        for filename in ("persona.json", "persona_introvert.json", "persona_extrovert.json"):
            path = CONFIG_DIR / filename
            config = PersonaConfig.from_file(path)
            assert config.name
            assert 0 <= config.big_five.openness <= 100
            assert config.mbti.type_code in MBTI_TYPE_DESCRIPTIONS
