import json

import numpy as np

BUCKET_BASE_URL = "https://storage.googleapis.com/catmatch/"


def load_image_paths() -> list[str]:
    with open("./image_paths.json", "r") as f:
        return json.load(f)


IMAGE_PATHS = load_image_paths()


def get_random_cat_from_breed(breed: str) -> str:
    breed_images = [
        BUCKET_BASE_URL + image_url
        for image_url in IMAGE_PATHS
        if breed.title() in image_url
    ]
    return np.random.choice(breed_images)


def get_all_cat_breeds() -> list[str]:
    breeds = []
    for image_url in IMAGE_PATHS:
        breed = image_url.split("/")[1]
        if breed not in breeds:
            breeds.append(breed)
    return breeds


def get_image_url_from_index(index: int):
    return BUCKET_BASE_URL + IMAGE_PATHS[index]


def get_index_from_image_url(image_url: str) -> int | None:
    try:
        return IMAGE_PATHS.index(image_url.replace(BUCKET_BASE_URL, ""))
    except ValueError:
        return None


def convert_ratings_dict_to_array(
    ratings_dict: dict[str, bool | None]
) -> np.ndarray | None:
    ratings_array = np.full(len(IMAGE_PATHS), np.nan)
    for key, value in ratings_dict.items():
        index = get_index_from_image_url(key)
        if index is None:
            return None
        ratings_array[index] = 1 if value else 0
    return ratings_array
