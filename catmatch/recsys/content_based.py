import dis
import math
from typing import NamedTuple

import h5py
import numpy as np


def read_h5py_file(path: str) -> np.ndarray:
    f = h5py.File(path, "r")
    matrix: np.ndarray = f["data"][()]  # type: ignore
    return matrix.astype(np.float64)


def _get_likeness_scores(
    user_ratings: np.ndarray, similarity_matrix: np.ndarray
) -> np.ndarray:
    indices_of_liked_items = np.where(user_ratings == 1)[0]
    indices_of_disliked_items = np.where(user_ratings == 0)[0]
    number_of_liked_items = len(indices_of_liked_items)
    number_of_disliked_items = len(indices_of_disliked_items)
    total_number = number_of_liked_items + number_of_disliked_items
    # Get the average similarity vector for the items the user has liked, a 1 * n array
    # where n is the number of items.
    similarity_matrix_f64 = similarity_matrix.astype(np.float64)
    liked_similarities_sum = similarity_matrix_f64[indices_of_liked_items].sum(axis=0)
    disliked_similarities_sum = similarity_matrix_f64[indices_of_disliked_items].sum(
        axis=0
    )
    # Subtract the disliked items similarity score from the liked items similarity score
    # Thus we have a vector that represents the how much the user likes each item
    # Add the maximum value of the disliked_similarities to the array
    # to avoid negative values
    weighted_like_scores = liked_similarities_sum * number_of_liked_items / total_number
    weighted_dislike_scores = (
        disliked_similarities_sum * number_of_disliked_items / total_number
    )
    likeness_scores = (
        weighted_like_scores - weighted_dislike_scores + np.max(weighted_dislike_scores)
    )

    return likeness_scores


def get_outliers(similarity_matrix: np.ndarray, threshold: float | None = None):
    threshold = threshold or (similarity_matrix.mean() - similarity_matrix.std())
    average_scores = similarity_matrix.mean(axis=0)
    outliers = np.where(average_scores < threshold)[0]
    return outliers


def recommend_k_new_items(
    user_ratings: np.ndarray,
    similarity_matrix: np.ndarray,
    outliers_indices: np.ndarray,
    k: int = 20,
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
    likeness_scores = _get_likeness_scores(user_ratings, similarity_matrix)

    # average_similarity_vector = similarity_matrix[indices_of_liked_items].mean(axis=0)
    # Disregard items the user has already seen
    likeness_scores[indices_of_seen_items] = -np.inf
    # Disregard outliers
    likeness_scores[outliers_indices] = -np.inf
    # Take the items the user has not seen
    # Get the top 80% of the items array with the highest values
    # Subtract the items the user has seen
    # half = len(likeness_scores) // 2
    cutoff_size = int(len(likeness_scores) * 0.2)
    cutoff_index = (
        len(likeness_scores)
        - len(indices_of_seen_items)
        - cutoff_size
        - len(outliers_indices)
    )
    top_indices = np.argpartition(likeness_scores, -cutoff_index)[-cutoff_index:]
    # Non-nan value indices
    non_nan_top_indices = np.where(
        ~np.isnan(likeness_scores[top_indices])
        & ~np.isinf(likeness_scores[top_indices])
    )[0]
    top_scores = likeness_scores[top_indices][non_nan_top_indices]
    # Convert the values to probabilities
    # This represents how likely the user is to like each item
    top_item_weights = top_scores / top_scores.sum()
    # Get a random sample of the top half of the items weighted by
    # how likely the user is to like the item
    return np.random.choice(top_indices, size=k, p=top_item_weights)


class MostAndLeastLiked(NamedTuple):
    most_liked_indices: np.ndarray
    least_liked_indices: np.ndarray


def get_most_and_least_liked_items2(
    user_ratings: np.ndarray,
    similarity_matrix: np.ndarray,
    outliers_indices: np.ndarray,
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
    likness_scores = _get_likeness_scores(user_ratings, similarity_matrix)
    # Get the average similarity vector for the items the user has disliked, a 1 * n
    # array, where n is the number of items.
    # Get the indices of the most and least liked items
    sorted_similarities = np.argsort(likness_scores)
    # Need to reverse for the most liked to come first
    most_liked_indices = sorted_similarities[::-1]
    least_liked_indices = sorted_similarities
    most_liked_indices = most_liked_indices[
        np.isin(most_liked_indices, outliers_indices, invert=True)
    ]
    least_liked_indices = least_liked_indices[
        np.isin(least_liked_indices, outliers_indices, invert=True)
    ]
    top_liked = most_liked_indices[:number_of_liked_items]
    top_disliked = least_liked_indices[:number_of_disliked_items]

    return MostAndLeastLiked(top_liked, top_disliked)


def get_most_and_least_liked_items(
    user_ratings: np.ndarray,
    similarity_matrix: np.ndarray,
    outliers_indices: np.ndarray,
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
    # Compute likeness scores (assuming _get_likeness_scores is defined elsewhere)
    likness_scores = _get_likeness_scores(user_ratings, similarity_matrix)

    # Create a mask for non-outlier indices
    non_outliers_mask = ~np.isin(np.arange(likness_scores.size), outliers_indices)

    # Filter out the scores of outliers
    filtered_scores = likness_scores[non_outliers_mask]
    filtered_indices = np.arange(likness_scores.size)[non_outliers_mask]

    # Find indices for top liked and disliked items without sorting the entire array
    # Get the partition indices for the top liked
    partition_index_liked = np.argpartition(
        -filtered_scores, number_of_liked_items - 1
    )[:number_of_liked_items]
    top_liked = filtered_indices[partition_index_liked]

    # Get the partition indices for the top disliked
    partition_index_disliked = np.argpartition(
        filtered_scores, number_of_disliked_items - 1
    )[:number_of_disliked_items]
    top_disliked = filtered_indices[partition_index_disliked]

    # Filter out outliers
    # print("most_liked_indices", top_liked)
    # print("most_liked_scores", likness_scores[top_liked])
    # print("least_liked_indices", least_liked_indices)
    # print("least_liked_scores", likness_scores[top_disliked])
    return MostAndLeastLiked(top_liked, top_disliked)
