"""FastAPI backend for Telegram Mini App."""
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from webapp.routes import settings, videos, export


def create_app(testing: bool = False) -> FastAPI:
    app = FastAPI(title="YouTube Summary Bot Dashboard")

    app.include_router(settings.router)
    app.include_router(videos.router)
    app.include_router(export.router)

    @app.get("/health")
    def health():
        return {"status": "ok"}

    return app


if __name__ == "__main__":
    import uvicorn
    app = create_app()
    uvicorn.run(app, host="0.0.0.0", port=8000)
