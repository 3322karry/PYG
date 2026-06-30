"""pyGal 全局配置。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class PyGalConfig:
    """pyGal 全局配置。"""
    # 项目根目录（MaiBot fork 根目录）
    project_root: Path = Path(".")

    # Agent 配置
    agent_max_loops: int = 10
    agent_use_checkpoint: bool = True

    # LLM 配置（对接 MaiBot 的模型管理器）
    llm_default_model: str = "default"
    llm_temperature: float = 0.8
    llm_max_tokens: int = 512

    # 调度器配置
    scheduler_check_interval: float = 60.0
    scheduler_boredom_threshold: float = 0.7

    # 人格配置文件路径
    persona_config_path: Path = Path("config/pygal/persona.json")

    # WebUI 配置
    webui_agent_trace_max: int = 100  # 最多保留多少条推理追踪

    def to_dict(self) -> dict:
        return {
            "agent_max_loops": self.agent_max_loops,
            "agent_use_checkpoint": self.agent_use_checkpoint,
            "llm_default_model": self.llm_default_model,
            "llm_temperature": self.llm_temperature,
            "llm_max_tokens": self.llm_max_tokens,
            "scheduler_check_interval": self.scheduler_check_interval,
            "scheduler_boredom_threshold": self.scheduler_boredom_threshold,
            "persona_config_path": str(self.persona_config_path),
            "webui_agent_trace_max": self.webui_agent_trace_max,
        }
