import asyncio
import logging

from src.bot import create_app

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)


def main():
    app = create_app()
    app.run_polling()


if __name__ == "__main__":
    main()
