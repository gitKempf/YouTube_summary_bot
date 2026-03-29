"""FastAPI backend for Telegram Mini App."""
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, FileResponse
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

    # Serve Quartz static site at /vault/ with .html fallback
    if not testing and QUARTZ_PUBLIC.exists():
        @app.get("/vault/{path:path}")
        async def serve_vault(path: str):
            file_path = QUARTZ_PUBLIC / path

            # Exact file exists (e.g. static assets, .css, .js)
            if file_path.is_file():
                return FileResponse(file_path)

            # Try appending .html (Quartz page without extension)
            html_path = file_path.with_suffix(".html")
            if html_path.is_file():
                return FileResponse(html_path)

            # Directory with index.html
            index_path = file_path / "index.html"
            if index_path.is_file():
                return FileResponse(index_path)

            # Fallback to 404 page or root
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
