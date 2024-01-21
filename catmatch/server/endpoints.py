import logging
import random
import re

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from catmatch.recsys.content_based import ContentBasedRecommender
from catmatch.server.utils import (
    IMAGE_PATHS,
    convert_ratings_dict_to_array,
    get_image_url_from_index,
)

recsys_router = APIRouter()


class RecommendationsBody(BaseModel):
    ratings: dict[str, bool | None]
    number_of_recommendations: int = 20


class RecommendationsResponse(BaseModel):
    recommendations: list[str]


recommender = ContentBasedRecommender(similarity_matrix_path="./similarity_matrix.hdf5")
logger = logging.getLogger(__name__)


@recsys_router.post("/recommendations")
async def recommendations(body: RecommendationsBody):
    if all(value is None for value in body.ratings.values()):
        # return random recommendations
        random_recomms = random.sample(IMAGE_PATHS, body.number_of_recommendations)
        return RecommendationsResponse(recommendations=random_recomms)

    ratings_array = convert_ratings_dict_to_array(body.ratings)
    if ratings_array is None:
        raise HTTPException(status_code=400, detail="Invalid image URL")
    # return random recommendations
    logger.info("Calculating recommendations...")
    recommendations = recommender.recommend_k_new_items(
        ratings_array, body.number_of_recommendations
    )
    recommendation_urls = [get_image_url_from_index(index) for index in recommendations]
    return RecommendationsResponse(recommendations=recommendation_urls)


@recsys_router.get("/hello")
async def hello():
    return "Hello!"
