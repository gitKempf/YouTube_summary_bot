"""FastAPI backend for Telegram Mini App."""
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from webapp.routes import settings, videos, export

QUARTZ_PUBLIC = Path(__file__).parent / "quartz" / "public"
TEMPLATES_DIR = Path(__file__).parent / "templates"


def create_app(testing: bool = False) -> FastAPI:
    app = FastAPI(title="YouTube Summary Bot Dashboard")

    app.include_router(settings.router)
    app.include_router(videos.router)
    app.include_router(export.router)

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/", response_class=HTMLResponse)
    def dashboard():
        html = (TEMPLATES_DIR / "settings.html").read_text()
        return HTMLResponse(content=html)

    # Serve Quartz static site at /vault/
    if not testing and QUARTZ_PUBLIC.exists():
        app.mount("/vault", StaticFiles(directory=str(QUARTZ_PUBLIC), html=True), name="vault")

    return app


if __name__ == "__main__":
    import uvicorn
    app = create_app()
    uvicorn.run(app, host="0.0.0.0", port=8000)
