import logging

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from sympy import O

from catmatch.recsys.content_based import (
    get_most_and_least_liked_items,
    read_h5py_file,
    recommend_k_new_items,
)
from catmatch.server.utils import (
    convert_ratings_dict_to_array,
    get_all_cat_breeds,
    get_image_url_from_index,
    get_random_cat_from_breed,
)

recsys_router = APIRouter()

logger = logging.getLogger(__name__)


def normalize_similarity_matrix(similarity_matrix: np.ndarray) -> np.ndarray:
    """Normalize the similarity matrix to be between 0 and 1"""
    arr_mean = similarity_matrix.mean()
    arr_std = similarity_matrix.std()

    normalized = (similarity_matrix - arr_mean) / arr_std
    nonzero_matrix = normalized + normalized.max()
    print("nonzero_matrix", nonzero_matrix)
    return nonzero_matrix


ALL_CAT_BREEDS = get_all_cat_breeds()
SIMILARITY_MATRIX = normalize_similarity_matrix(
    read_h5py_file("./similarity_matrix.hdf5")
)


def get_initial_recommendations():
    initial_cats = []
    for breed in ALL_CAT_BREEDS:
        initial_cats.append(get_random_cat_from_breed(breed))
    return initial_cats


class RecommendationsBody(BaseModel):
    ratings: dict[str, bool | None]
    number_of_recommendations: int = 20


class RecommendationsResponse(BaseModel):
    recommendations: list[str]


@recsys_router.post("/recommendations")
async def recommendations(body: RecommendationsBody):
    # -- Initial recommendations --
    rating_count = len(body.ratings)
    if rating_count < len(ALL_CAT_BREEDS):
        # guarantee the first recommendations are from one of each breed
        initial = get_initial_recommendations()
        if len(initial) >= body.number_of_recommendations - rating_count:
            return RecommendationsResponse(
                recommendations=initial[
                    rating_count : rating_count + body.number_of_recommendations
                ]
            )

        # if we don't have enough initial recommendations,
        # fill the rest with actual recommendations
        missing_recommendations = body.number_of_recommendations - len(initial)
        ratings_array = convert_ratings_dict_to_array(body.ratings)
        if ratings_array is None:
            raise HTTPException(status_code=400, detail="Invalid image URL")
        addon_recoms = recommend_k_new_items(
            ratings_array, SIMILARITY_MATRIX, missing_recommendations
        )
        total_recoms_indices = initial + list(addon_recoms)
        total_recom_urls = [
            get_image_url_from_index(index) for index in total_recoms_indices
        ]
        return RecommendationsResponse(recommendations=total_recom_urls)

    # -- Normal recommendations --
    ratings_array = convert_ratings_dict_to_array(body.ratings)
    if ratings_array is None:
        raise HTTPException(status_code=400, detail="Invalid image URL")
    addon_recoms = recommend_k_new_items(
        ratings_array, SIMILARITY_MATRIX, body.number_of_recommendations
    )
    recommendation_urls = [get_image_url_from_index(index) for index in addon_recoms]
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
        SIMILARITY_MATRIX,
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
