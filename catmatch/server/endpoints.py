import logging
import random

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from catmatch.recsys.content_based import (
    get_most_and_least_liked_items,
    read_h5py_file,
    recommend_k_new_items,
)
from catmatch.server.utils import (
    BUCKET_BASE_URL,
    IMAGE_PATHS,
    convert_ratings_dict_to_array,
    get_image_url_from_index,
)

recsys_router = APIRouter()

similarity_matrix = read_h5py_file("./similarity_matrix.hdf5")
logger = logging.getLogger(__name__)


class RecommendationsBody(BaseModel):
    ratings: dict[str, bool | None]
    number_of_recommendations: int = 20


class RecommendationsResponse(BaseModel):
    recommendations: list[str]


@recsys_router.post("/recommendations")
async def recommendations(body: RecommendationsBody):
    if all(value is None for value in body.ratings.values()):
        # return random recommendations
        random_recomms_endings = random.sample(
            IMAGE_PATHS, body.number_of_recommendations
        )
        random_cat_urls = [
            BUCKET_BASE_URL + ending for ending in random_recomms_endings
        ]
        return RecommendationsResponse(recommendations=random_cat_urls)

    ratings_array = convert_ratings_dict_to_array(body.ratings)
    if ratings_array is None:
        raise HTTPException(status_code=400, detail="Invalid image URL")
    # return random recommendations
    recommendations = recommend_k_new_items(
        ratings_array, similarity_matrix, body.number_of_recommendations
    )
    recommendation_urls = [get_image_url_from_index(index) for index in recommendations]
    return RecommendationsResponse(recommendations=recommendation_urls)


class MostLeastLikedBody(BaseModel):
    ratings: dict[str, bool | None]
    number_of_liked_items: int = 3
    number_of_disliked_items: int = 3


class MostLeastLikedResponse(BaseModel):
    most_liked: list[str]
    least_liked: list[str]


@recsys_router.post("/most-least-liked")
async def most_and_least_liked(body: MostLeastLikedBody):
    ratings_array = convert_ratings_dict_to_array(body.ratings)
    if ratings_array is None:
        raise HTTPException(status_code=400, detail="Invalid image URL")
    # return random recommendations
    most_liked_items, least_liked_items = get_most_and_least_liked_items(
        ratings_array,
        similarity_matrix,
        body.number_of_liked_items,
        body.number_of_disliked_items,
    )
    most_liked_urls = [get_image_url_from_index(index) for index in most_liked_items]
    least_liked_urls = [get_image_url_from_index(index) for index in least_liked_items]
    return MostLeastLikedResponse(
        most_liked=most_liked_urls, least_liked=least_liked_urls
    )


@recsys_router.get("/hello")
async def hello():
    return "Hello!"
