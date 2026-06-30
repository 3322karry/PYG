# pyGal — Pygmalion yields Galatea

> 从皮格马利翁的雕像到活生生的伽拉忒亚，从被动机器人到全真虚拟网友。

pyGal 是 [MaiBot](https://github.com/MaiM-with-u/MaiBot) 的扩展模块，引入 LangGraph Agent 框架，将 MaiBot 改造为具有主动行动力、能自我进化、人格丰满的全真虚拟网友。

## ✨ 特性

- **LangGraph 状态图引擎** — 五节点循环（感知→反思→决策→行动→观察），替换 Maisaka 推理层
- **大五人格 + MBTI** — 可量化的人格模型，数值映射到 8 个行为参数，动态渲染 System Prompt
- **主动行动系统** — 精力值/无聊度/社交驱动，消息事件 + 定时器双触发，概率决策 + LLM CoT 评估
- **Skill 调用系统** — 标准化 Tool 抽象（SearchWeb / GetTime / QueryLPMM），动态启/禁
- **自我反思** — 深度对话后自动触发反思，LLM 提炼记忆节点，主动写入 A-Memorix LPMM
- **WebUI** — 人格滑块 / 技能管理 / Agent 推理链追踪 / 反思历史 / 初始化向导

## 📦 安装

pyGal 已集成在 MaiBot fork 中，安装 MaiBot 依赖时自动安装：

```bash
git clone https://github.com/<your-username>/MaiBot.git
cd MaiBot
uv sync  # 或 pip install -e .
```

## 🚀 使用

### 独立测试模式（Mock）

不需要 MaiBot 运行时环境，使用 Mock 适配器：

```bash
# 运行 155 项测试
PYTHONPATH=src python -m pytest tests/pygal/ -q

# 启动 WebUI
python -c "
import asyncio
from pygal import init
state = asyncio.run(init(use_mock=True))
from pygal.webui.app import create_app
import uvicorn
app = create_app(state=state)
app.state.pygal = state
uvicorn.run(app, host='0.0.0.0', port=8002)
"
```

### 集成 MaiBot 运行

在 MaiBot 启动后调用 pyGal 初始化：

```python
from pygal import init as init_pygal

# MaiBot 启动流程中
pygal_state = await init_pygal(
    persona_config_path="config/pygal/persona.json",
    use_mock=False,  # 使用真实 MaiBot 适配器
)
```

WebUI 集成到 MaiBot 的 FastAPI 应用：

```python
from pygal.webui.app import create_app
# 将 pyGal 的路由挂载到 MaiBot WebUI
pygal_app = create_app(state=pygal_state)
```

### WebUI 访问

启动后访问 `http://localhost:8002`：

- **配置页** — 大五人格滑块 + MBTI 选择 + 实时预览
- **主动行为与状态** — 精力/无聊度实时显示 + 调度器配置
- **技能管理** — 工具开关 + 调用历史
- **Agent 追踪** — 推理链时间线 + 反思记录
- **初始化向导** — API Key + 人格捏造 + 技能选择

## 🏗️ 架构

```
pygal/
├── core/           # LangGraph Agent 状态图（五节点循环）
│   ├── state.py        # PyGalState 共享状态
│   ├── nodes.py        # perceive/reflect/decide/act/observe
│   └── graph.py        # StateGraph 构建 + PyGalAgent
├── adapter/        # MaiBot 适配层
│   ├── llm.py          # MaiBotLLMAdapter → llm_models
│   ├── lpmm.py         # MaiBotLPMMAdapter → A-Memorix
│   └── platform.py     # MaiBotPlatformAdapter → PlatformIO
├── persona/        # 人格引擎
│   ├── model.py        # BigFive + MBTI + PersonaConfig
│   └── renderer.py     # 数值→行为参数→System Prompt
├── scheduler/      # 主动行动调度器
├── tools/          # Skill 系统
│   ├── base.py         # Tool/ToolRegistry/ToolResult
│   ├── builtin.py      # SearchWeb/GetTime/QueryLPMM
│   └── manager.py      # ToolManager
├── reflection/     # 反思引擎
│   ├── models.py       # MemoryNode/ImpressionUpdate
│   └── engine.py       # LLM 反思→结构化→写入 LPMM
├── webui/          # FastAPI + SPA 前端
│   ├── app.py          # 应用工厂
│   ├── routers/        # API 路由
│   └── static/         # 前端页面
└── init.py         # 集成入口
```

## 📝 配置

人格配置文件位于 `config/pygal/persona.json`：

```json
{
  "name": "Galatea",
  "nickname": "伽拉",
  "big_five": {
    "openness": 75,
    "conscientiousness": 35,
    "extraversion": 65,
    "agreeableness": 70,
    "neuroticism": 55
  },
  "mbti": { "ei": "E", "sn": "N", "tf": "F", "jp": "P" },
  "background": "一个喜欢深夜冲浪的网友",
  "interests": ["编程", "游戏", "猫咪"]
}
```

预设角色：`persona.json`（伽拉）/ `persona_introvert.json`（冷锋）/ `persona_extrovert.json`（小鹿）

## 📄 License

GPL-3.0（继承 MaiBot）
