"""FastAPI backend for Telegram Mini App."""
import os
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse

from webapp.routes import settings, videos, export, knowledge
from webapp.auth import validate_init_data, TelegramAuthError

TEMPLATES_DIR = Path(__file__).parent / "templates"
BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")


def create_app(testing: bool = False) -> FastAPI:
    app = FastAPI(title="YouTube Summary Bot Dashboard")

    # Protect /api/ endpoints with Telegram auth (skip in testing)
    if not testing and BOT_TOKEN:
        from starlette.middleware.base import BaseHTTPMiddleware
        from starlette.responses import JSONResponse

        class TelegramAuthMiddleware(BaseHTTPMiddleware):
            async def dispatch(self, request: Request, call_next):
                if request.url.path.startswith("/api/"):
                    init_data = request.headers.get("x-telegram-init-data", "")
                    if not init_data:
                        return JSONResponse({"detail": "Telegram authentication required"}, 401)
                    try:
                        validate_init_data(init_data, BOT_TOKEN)
                    except TelegramAuthError as e:
                        return JSONResponse({"detail": str(e)}, 401)
                return await call_next(request)

        app.add_middleware(TelegramAuthMiddleware)

    app.include_router(settings.router)
    app.include_router(videos.router)
    app.include_router(export.router)
    app.include_router(knowledge.router)

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/", response_class=HTMLResponse)
    def landing():
        html = (TEMPLATES_DIR / "landing.html").read_text()
        return HTMLResponse(content=html)

    @app.get("/app", response_class=HTMLResponse)
    def dashboard():
        html = (TEMPLATES_DIR / "settings.html").read_text()
        return HTMLResponse(content=html)

    return app


if __name__ == "__main__":
    import uvicorn
    app = create_app()
    uvicorn.run(app, host="0.0.0.0", port=8000)
