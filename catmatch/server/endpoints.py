from fastapi import APIRouter

recsys_router = APIRouter()


@recsys_router.get("/hello")
async def root():
    return {"message": "Hello World"}
