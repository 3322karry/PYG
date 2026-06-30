"""pyGal WebUI 应用工厂 — FastAPI 实例创建与路由注册。"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

from .routers import (
    persona_router,
    skills_router,
    agent_trace_router,
    init_guide_router,
)
from .state import WebUIState


def create_app(
    state: Optional[WebUIState] = None,
    host: str = "0.0.0.0",
    port: int = 8002,
) -> FastAPI:
    """创建 pyGal WebUI FastAPI 应用。

    Args:
        state: WebUI 共享状态（持有 Agent / ToolManager / ReflectionEngine 引用）
        host: 监听地址
        port: 监听端口

    Returns:
        FastAPI 应用实例
    """
    app = FastAPI(title="pyGal WebUI", version="0.1.0")

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:5173",
            "http://127.0.0.1:5173",
            f"http://localhost:{port}",
            f"http://127.0.0.1:{port}",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 共享状态
    app.state.pygal = state or WebUIState()

    # 注册 API 路由
    app.include_router(persona_router, prefix="/api/persona", tags=["persona"])
    app.include_router(skills_router, prefix="/api/skills", tags=["skills"])
    app.include_router(agent_trace_router, prefix="/api/agent_trace", tags=["agent_trace"])
    app.include_router(init_guide_router, prefix="/api/init_guide", tags=["init_guide"])

    # 健康检查
    @app.get("/health")
    async def health():
        return {"status": "ok", "service": "pyGal WebUI"}

    # 静态文件（前端 SPA）
    static_dir = Path(__file__).parent / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

        @app.get("/{full_path:path}")
        async def spa(full_path: str):
            """SPA fallback — 所有非 API 路由返回 index.html。"""
            file_path = static_dir / full_path
            if file_path.exists() and file_path.is_file():
                return FileResponse(str(file_path))
            return FileResponse(str(static_dir / "index.html"))

    return app
