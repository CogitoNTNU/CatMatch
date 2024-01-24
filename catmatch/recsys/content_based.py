from typing import NamedTuple

import h5py
import numpy as np


def read_h5py_file(path: str) -> np.ndarray:
    f = h5py.File(path, "r")
    return f["data"][()]  # type: ignore


def recommend_k_new_items(
    user_ratings: np.ndarray, similarity_matrix: np.ndarray, k: int = 20
) -> np.ndarray:
    """Recommend k new items to the user based on their rated items
    and a similarity matrix between all items.

    Args:
        user_ratings (np.ndarray): The array of ratings the user has given.
            Is NaN for unrated items, 1 for liked, and 0 for disliked items.
        similarity_matrix (np.ndarray): The similarity matrix between all items,
            is a matrix of size n x n.
        k (int, optional): The number of recommendations for the. Defaults to 20.

    Returns:
        np.ndarray: The indices of items in from the similarity matrix
            of the k best items to recommend.
            These are a set of random items from the top half of the items
            the user are most likely to like.
    """
    # Find the K most similar cats to the user's cats
    # 1. Take the cosine similarity vector for each cat the user has rated,
    #    then take the average of these vectors. Then choose randomly among the items
    #    with the highest values in this vecto.
    indices_of_seen_items = np.where(~np.isnan(user_ratings))[0]
    indices_of_liked_items = np.where(user_ratings == 1)[0]
    # Get the average similarity vector for the items the user has liked, a 1 * n array
    # where n is the number of items.
    average_similarity_vector = similarity_matrix[indices_of_liked_items].mean(axis=0)
    # Don't recommend the items the user has already seen
    average_similarity_vector[indices_of_seen_items] = -np.inf
    # Take the top half of the array with the highest values
    # Get half of the array with the highest values
    half = len(average_similarity_vector) // 2
    top_half_indices = np.argpartition(average_similarity_vector, -half)[-half:]
    top_half_similarities = average_similarity_vector[top_half_indices]
    # Convert the values to probabilities
    # This represents how likely the user is to like each item
    top_half_items_weights = top_half_similarities / top_half_similarities.sum()
    # Get a random sample of the top half of the items weighted by
    # how likely the user is to like the item
    return np.random.choice(top_half_indices, size=k, p=top_half_items_weights)


class MostAndLeastLiked(NamedTuple):
    most_liked_indices: np.ndarray
    least_liked_indices: np.ndarray


def get_most_and_least_liked_items(
    user_ratings: np.ndarray,
    similarity_matrix: np.ndarray,
    number_of_liked_items: int = 3,
    number_of_disliked_items: int = 3,
) -> MostAndLeastLiked:
    """Get the k most and least liked items for a user.

    Args:
        user_ratings (np.ndarray): The array of ratings the user has given.
            Is NaN for unrated items, 1 for liked, and 0 for disliked items.
        similarity_matrix (np.ndarray): The similarity matrix between all items,
            is a matrix of size n x n.
        k (int, optional): The number of most and least liked items to return.
            Defaults to 3.

    Returns:
        tuple[np.ndarray, np.ndarray]: The indices of the most and least liked items
            for the user.
    """
    # Get the indices of the items the user has liked and disliked
    indices_of_liked_items = np.where(user_ratings == 1)[0]
    # Get the average similarity vector for the items the user has liked, a 1 * n array
    # where n is the number of items.
    average_similarity_vector = similarity_matrix[indices_of_liked_items].mean(axis=0)
    # Get the average similarity vector for the items the user has disliked, a 1 * n
    # array
    # where n is the number of items.
    # Get the indices of the most and least liked items
    sorted_similarities = np.argsort(average_similarity_vector)
    # Need to reverse for the most liked to come first
    most_liked_indices = sorted_similarities[-number_of_liked_items:][::-1]
    least_liked_indices = sorted_similarities[:number_of_disliked_items]
    return MostAndLeastLiked(most_liked_indices, least_liked_indices)
