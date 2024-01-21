import random

import h5py
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def read_h5py_file(path: str) -> np.ndarray:
    f = h5py.File(path, "r")
    return f["data"][()]  # type: ignore


class ContentBasedRecommender:
    def __init__(
        self,
        similarity_matrix_path: str = "./similarity_matrix.hdf5",
    ):
        self.similarity_matrix = read_h5py_file(similarity_matrix_path)

    def recommend_k_new_items(self, user_ratings: np.ndarray, k: int = 20):
        # Find the K most similar cats to the user's cats
        # There are two ways to do this:
        # 1. Take the average of the embedding of all the cats the user has rated,
        #   then compute the cosine similarity between this average embedding and all the other cats.
        #   then take the K most similar cats (highest values).
        # 2. Take the cosine similarity vector for each cat the user has rated,
        #    then take the average of these vectors. Then choose the items
        #    with the k highest values in this vector.
        indices_of_rated_items = np.where(~np.isnan(user_ratings))[0]
        # similarity_vectors = self.similarity_matrix[indices]
        average_similarity_vectors = self.similarity_matrix[
            indices_of_rated_items
        ].mean(axis=0)
        return np.argsort(average_similarity_vectors)[::-1][:k]

    # def recommend_k_movies(self, userId, k):
    #     return self.user_ratings.loc[userId].sort_values(ascending=False)[:k]

    # def recommend_k_new_movies(self, userId, k):
    #     return (self.user_ratings.loc[userId] * self.cats_seen.loc[userId]).sort_values(
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
