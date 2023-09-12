import logging

from fastapi import FastAPI

from catmatch.server.endpoints import recsys_router

app = FastAPI()

logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(level=logging.DEBUG)
    logger.info("Hello world!")
    app.include_router(recsys_router, prefix="/recsys", tags=["recsys"])


main()
