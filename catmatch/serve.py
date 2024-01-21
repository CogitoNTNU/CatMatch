import logging

from fastapi import FastAPI

from catmatch.server.endpoints import recsys_router

app = FastAPI()

logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    app.include_router(recsys_router, tags=["recsys"])


main()
