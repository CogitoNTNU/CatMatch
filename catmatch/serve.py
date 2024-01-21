import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from catmatch.server.endpoints import recsys_router

app = FastAPI()

logger = logging.getLogger(__name__)

app = FastAPI()

origins = [
    "catmatch.rosby.no",
    "http://localhost",
    "http://localhost:8080",
    "http://localhost:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    app.include_router(recsys_router, tags=["recsys"])


main()
