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
    """
    # Find the K most similar cats to the user's cats
    # 2. Take the cosine similarity vector for each cat the user has rated,
    #    then take the average of these vectors. Then choose the items
    #    with the k highest values in this vector.
    indices_of_seen_items = np.where(~np.isnan(user_ratings))[0]
    indices_of_liked_items = np.where(user_ratings == 1)[0]
    # Get the average similarity vector for the items the user has liked, a 1 * n array
    # where n is the number of items.
    average_similarity_vector = similarity_matrix[indices_of_liked_items].mean(axis=0)
    # Don't recommend the items the user has already seen
    average_similarity_vector[indices_of_seen_items] = -np.inf
    # Take the top half of the array with the highest values
    half = len(average_similarity_vector) // 2
    # Get half of the array with the highest values
    top_half_indices = np.argpartition(average_similarity_vector, -half)[-half:]
    top_half_similarities = average_similarity_vector[top_half_indices]
    # Get a random sample of the top half weighted by their similarity
    # (higher similarity = higher chance of being sampled)
    item_weights = top_half_similarities / top_half_similarities.sum()
    return np.random.choice(top_half_indices, size=k, p=item_weights)

    # def recommend_k_movies(self, userId, k):
    #     return self.user_ratings.loc[userId].sort_values(ascending=False)[:k]

    # def recommend_k_new_movies(self, userId, k):
    #     return (self.user_ratings.loc[userId] * self.cats_seen.loc[userId])
    #   .sort_values(
    #         ascending=False
    #     )[:k]


# class ContentBasedRecommenderCats:
#     def __init__(self, item_similarities: np.ndarray, kmost: int = 20) -> None:
#         self.item_similarities = item_similarities
#         self.kmost = kmost

#     def fit(self, matrix_train: np.ndarray):
#         self.rating_matrix = matrix_train

#     def _predict_rating(self, user_document_similarities, user_ratings) -> float:
#         k_most_similar_document_ratings = user_ratings[
#             np.argpartition(user_document_similarities, -self.kmost, axis=1)[
#                 ::, -self.kmost :
#             ]
#         ]
#         # Return average/median rating of the k most similar documents (ignoring NaNs)
#         return np.nanmean(k_most_similar_document_ratings, axis=1)

#     def predict_rating(self, user_index: int, document_index: int) -> float:
#         return self._predict_rating(
#             self.item_similarities.copy(), self.rating_matrix[user_index, :].copy()
#         )[document_index]

#     def predict_ratings_for_user(self, user_index: int) -> np.ndarray:
#         user_ratings = self.rating_matrix[user_index, :]
#         user_document_similarities = self.item_similarities.copy()
#         user_document_similarities[:, np.isnan(user_ratings)] = -np.inf
#         predicted = self._predict_rating(user_document_similarities, user_ratings)
#         return predicted

#     def predict_all(self) -> np.ndarray:
#         ratings = np.ndarray(self.rating_matrix.shape)
#         for u in trange(self.rating_matrix.shape[0]):
#             ratings[u] = self.predict_ratings_for_user(u)
#         return ratings
