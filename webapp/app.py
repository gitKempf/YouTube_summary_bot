"""FastAPI backend for Telegram Mini App."""
import os
from pathlib import Path

from fastapi import FastAPI, Request, Header, HTTPException
from fastapi.responses import HTMLResponse, FileResponse

from webapp.routes import settings, videos, export
from webapp.auth import validate_init_data, TelegramAuthError

QUARTZ_PUBLIC = Path(__file__).parent / "quartz" / "public"
TEMPLATES_DIR = Path(__file__).parent / "templates"
BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")


def _check_telegram_auth(x_telegram_init_data: str = Header(None)):
    """Dependency: validate Telegram initData from header."""
    if not x_telegram_init_data or not BOT_TOKEN:
        raise HTTPException(status_code=401, detail="Telegram authentication required")
    try:
        return validate_init_data(x_telegram_init_data, BOT_TOKEN)
    except TelegramAuthError as e:
        raise HTTPException(status_code=401, detail=str(e))


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

    # Serve Quartz static site at /vault/ with .html fallback
    if not testing and QUARTZ_PUBLIC.exists():
        @app.get("/vault/{path:path}")
        async def serve_vault(path: str):
            file_path = QUARTZ_PUBLIC / path

            if file_path.is_file():
                return FileResponse(file_path)

            html_path = file_path.with_suffix(".html")
            if html_path.is_file():
                return FileResponse(html_path)

            index_path = file_path / "index.html"
            if index_path.is_file():
                return FileResponse(index_path)

            not_found = QUARTZ_PUBLIC / "404.html"
            if not_found.is_file():
                return FileResponse(not_found, status_code=404)
            return HTMLResponse("Not found", status_code=404)

        @app.get("/vault")
        @app.get("/vault/")
        async def vault_root():
            index = QUARTZ_PUBLIC / "index.html"
            if index.is_file():
                return FileResponse(index)
            return HTMLResponse("Vault not built yet", status_code=404)

    return app


if __name__ == "__main__":
    import uvicorn
    app = create_app()
    uvicorn.run(app, host="0.0.0.0", port=8000)
