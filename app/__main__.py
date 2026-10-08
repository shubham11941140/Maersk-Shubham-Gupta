"""``python -m app`` — run the API with JSON logging configured (no uvicorn access log)."""

import uvicorn

from app.config import get_settings


def main() -> None:
    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        log_config=None,
        access_log=False,
        proxy_headers=True,
    )


if __name__ == "__main__":
    main()
