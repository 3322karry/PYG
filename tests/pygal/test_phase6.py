"""pyGal 阶段 6 测试 — WebUI API。"""
from __future__ import annotations

import pytest
from pathlib import Path
from fastapi.testclient import TestClient

from pygal.webui.app import create_app
from pygal.webui.state import WebUIState
from pygal.adapter.llm import MockLLMAdapter
from pygal.adapter.lpmm import MockLPMMAdapter
from pygal.adapter.platform import MockPlatformAdapter
from pygal.persona.model import PersonaConfig, BigFive, MBTI
from pygal.persona.renderer import PersonaRenderer
from pygal.tools.manager import ToolManager
from pygal.reflection.engine import ReflectionEngine
from pygal.core.nodes import AgentNodes
from pygal.core.graph import PyGalAgent
from pygal.core.state import PyGalState, MessageEvent
from pygal.scheduler import ActionScheduler, SchedulerConfig


@pytest.fixture
def webui_state(tmp_path):
    """创建带完整组件的 WebUI 状态。"""
    lpmm = MockLPMMAdapter()
    llm = MockLLMAdapter()
    platform = MockPlatformAdapter()
    renderer = PersonaRenderer()
    tm = ToolManager(lpmm_adapter=lpmm)
    engine = ReflectionEngine(llm=llm, lpmm=lpmm)
    nodes = AgentNodes(
        llm=llm, lpmm=lpmm, platform=platform,
        persona_renderer=renderer, tool_manager=tm,
        reflection_engine=engine,
    )
    agent = PyGalAgent(nodes)
    scheduler = ActionScheduler(SchedulerConfig(message_debounce=0))

    state = WebUIState(
        agent=agent,
        tool_manager=tm,
        reflection_engine=engine,
        scheduler=scheduler,
        persona_config_path=tmp_path / "persona.json",
    )
    return state


@pytest.fixture
def client(webui_state):
    """创建测试客户端。"""
    app = create_app(webui_state)
    app.state.pygal = webui_state
    return TestClient(app)


# ── 1. 健康检查 ──

class TestHealth:

    def test_health(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "ok"
        assert data["service"] == "pyGal WebUI"


# ── 2. 人格 API ──

class TestPersonaAPI:

    def test_get_persona_default(self, client):
        """无配置时返回默认人格。"""
        r = client.get("/api/persona")
        assert r.status_code == 200
        data = r.json()
        assert "persona" in data
        assert data["persona"]["name"] == "Galatea"

    def test_save_and_get_persona(self, client):
        """保存后读取人格配置。"""
        payload = {
            "name": "测试角色",
            "nickname": "小测",
            "big_five": {"openness": 80, "conscientiousness": 30, "extraversion": 60, "agreeableness": 70, "neuroticism": 40},
            "mbti": {"ei": "E", "sn": "N", "tf": "T", "jp": "J"},
            "background": "测试背景",
            "interests": ["编程", "测试"],
        }
        r = client.post("/api/persona", json=payload)
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

        # 再读取
        r = client.get("/api/persona")
        data = r.json()
        assert data["persona"]["name"] == "测试角色"
        assert data["persona"]["big_five"]["openness"] == 80
        assert data["preview"]["action_tendency"] > 0.5

    def test_preview_persona(self, client):
        """实时预览不保存。"""
        payload = {
            "name": "预览角色",
            "nickname": "预览",
            "big_five": {"openness": 90, "conscientiousness": 50, "extraversion": 90, "agreeableness": 50, "neuroticism": 50},
            "mbti": {"ei": "E", "sn": "N", "tf": "F", "jp": "P"},
        }
        r = client.post("/api/persona/preview", json=payload)
        assert r.status_code == 200
        data = r.json()
        assert data["action_tendency"] > 0.8
        assert "system_prompt" in data

    def test_preview_reflects_slider_changes(self, client):
        """滑块变化反映到预览。"""
        low = client.post("/api/persona/preview", json={
            "name": "t", "big_five": {"openness": 10, "conscientiousness": 10, "extraversion": 10, "agreeableness": 10, "neuroticism": 10},
            "mbti": {"ei": "I", "sn": "S", "tf": "T", "jp": "J"},
        }).json()
        high = client.post("/api/persona/preview", json={
            "name": "t", "big_five": {"openness": 90, "conscientiousness": 90, "extraversion": 90, "agreeableness": 90, "neuroticism": 90},
            "mbti": {"ei": "E", "sn": "N", "tf": "F", "jp": "P"},
        }).json()
        assert high["action_tendency"] > low["action_tendency"]
        assert high["emoji_frequency"] > low["emoji_frequency"]


# ── 3. 技能 API ──

class TestSkillsAPI:

    def test_list_skills(self, client):
        r = client.get("/api/skills")
        assert r.status_code == 200
        skills = r.json()["skills"]
        assert len(skills) == 3
        names = [s["name"] for s in skills]
        assert "search_web" in names
        assert "get_time" in names
        assert "query_lpmm" in names

    def test_toggle_skill(self, client):
        r = client.put("/api/skills/search_web", json={"name": "search_web", "enabled": False})
        assert r.status_code == 200
        assert r.json()["enabled"] is False

        # 验证已禁用
        r = client.get("/api/skills")
        search = [s for s in r.json()["skills"] if s["name"] == "search_web"][0]
        assert search["enabled"] is False

    def test_toggle_nonexistent(self, client):
        r = client.put("/api/skills/nonexistent", json={"name": "nonexistent", "enabled": True})
        assert r.status_code == 200
        assert r.json()["status"] == "error"

    def test_skill_history(self, client):
        # 执行一次工具调用
        client.app.state.pygal.tool_manager.execute("get_time")
        r = client.get("/api/skills/history")
        assert r.status_code == 200
        history = r.json()["history"]
        assert len(history) >= 1
        assert history[0]["tool"] == "get_time"


# ── 4. Agent 追踪 API ──

class TestAgentTraceAPI:

    def test_get_trace_empty(self, client):
        r = client.get("/api/agent_trace")
        assert r.status_code == 200
        data = r.json()
        assert data["trace"] == []

    def test_get_trace_after_run(self, client):
        """Agent 运行后追踪有数据。"""
        import random
        random.seed(42)
        state = client.app.state.pygal
        # 运行 Agent
        agent_state = PyGalState()
        config = PersonaConfig(big_five=BigFive(extraversion=80, agreeableness=70), mbti=MBTI(ei="E"))
        agent_state.persona = PersonaRenderer().render(config)
        agent_state.messages = [MessageEvent(sender="u1", sender_name="Test", content="你好", chat_id="g1", is_mention_me=True)]
        result = state.agent.run(agent_state)
        state.update_after_agent_run(result)

        r = client.get("/api/agent_trace")
        data = r.json()
        assert len(data["trace"]) > 0
        assert data["action_result"] is not None

    def test_get_internal_state(self, client):
        r = client.get("/api/agent_trace/state")
        assert r.status_code == 200
        data = r.json()
        assert "internal" in data
        assert "initialized" in data

    def test_get_reflection_history(self, client):
        r = client.get("/api/agent_trace/reflection")
        assert r.status_code == 200
        assert "history" in r.json()


# ── 5. 初始化向导 API ──

class TestInitGuideAPI:

    def test_init_status(self, client):
        r = client.get("/api/init_guide/status")
        assert r.status_code == 200
        assert r.json()["initialized"] is False

    def test_submit_init(self, client):
        payload = {
            "api_key": "sk-test-key",
            "api_base": "https://api.deepseek.com/v1",
            "model_name": "deepseek-chat",
            "name": "伽拉",
            "nickname": "加拉",
            "background": "深夜网友",
            "interests": ["编程", "游戏"],
            "big_five": {"openness": 75, "conscientiousness": 35, "extraversion": 65, "agreeableness": 70, "neuroticism": 55},
            "mbti": {"ei": "E", "sn": "N", "tf": "F", "jp": "P"},
            "enabled_skills": ["search_web", "get_time"],
        }
        r = client.post("/api/init_guide", json=payload)
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "ok"
        assert data["api_configured"] is True
        assert "search_web" in data["enabled_skills"]

        # 验证已初始化
        r = client.get("/api/init_guide/status")
        assert r.json()["initialized"] is True

        # 验证人格已保存
        r = client.get("/api/persona")
        assert r.json()["persona"]["name"] == "伽拉"

        # 验证技能开关
        r = client.get("/api/skills")
        skills = {s["name"]: s["enabled"] for s in r.json()["skills"]}
        assert skills["search_web"] is True
        assert skills["query_lpmm"] is False  # 未选


# ── 6. 前端页面 ──

class TestFrontend:

    def test_index_html_served(self, client):
        r = client.get("/")
        assert r.status_code == 200
        assert "pyGal" in r.text
        assert "大五人格" in r.text

    def test_spa_fallback(self, client):
        r = client.get("/some-unknown-route")
        assert r.status_code == 200
        assert "pyGal" in r.text
