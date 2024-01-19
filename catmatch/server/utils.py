import pickle

import numpy as np

BASE_URL = "https://storage.googleapis.com/catmatch/"


def load_image_paths() -> list[str]:
    with open("./image_paths.pkl", "rb") as f:
        return pickle.load(f)


IMAGE_PATHS = load_image_paths()


def get_image_url_from_index(index: int):
    return BASE_URL + IMAGE_PATHS[index]


def get_index_from_image_url(image_url: str) -> int | None:
    try:
        return IMAGE_PATHS.index(image_url.replace(BASE_URL, ""))
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
        ratings_array[index] = 0 if value else 1
    return ratings_array
