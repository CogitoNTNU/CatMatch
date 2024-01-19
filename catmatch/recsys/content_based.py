import random

import h5py
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import trange


def read_h5py_file(path: str) -> np.ndarray:
    f = h5py.File(path, "r")
    return f["data"][()]  # type: ignore


class ContentBasedRecommender:
    def __init__(
        self,
        similarity_matrix_path: str = "./similarity_matrix.hdf5",
    ):
        # self.embedding = embedding
        self.similarity_matrix = read_h5py_file(similarity_matrix_path)
        # self.user_ratings = pd.DataFrame()  # TODO: remove
        # self.cats_seen = self.user_ratings.isna()

    # def matrix_pairwise_cosine_similarity(self):
    #     self.similarity_matrix = pd.DataFrame(
    #         cosine_similarity(self.embedding), index=self.embedding.index
    #     )
    #     self.similarity_matrix = self.similarity_matrix.T.set_index(
    #         self.embedding.index
    #     ).T

    # find the k most similar items to the item X
    # def k_most_similar_item_user(self, movie, userId):
    #     # softmax = lambda x: np.exp(x) / sum(np.exp(x))

    #     movies_rated = self.user_ratings.dropna()
    #     ind = movies_rated.index.tolist()
    #     if movie in ind:
    #         return movies_rated.loc[movie]
    #     else:
    #         ind.append(movie)
    #     # matrix = self.matrix_pairwise_cosine_similarity()

    #     matrix = pd.DataFrame()  # TODO: change
    #     new_matrix = matrix.loc[ind].T.loc[ind]
    #     k_best = new_matrix.loc[movie]
    #     k_best = k_best[~k_best.index.duplicated(keep="first")]
    #     # k_best = k_best[~k_best.columns.duplicated(keep='first')]
    #     k_best = k_best.sort_values(ascending=False)
    #     k_best.drop(movie, inplace=True)

    #     mat = pd.concat([(k_best), movies_rated], axis=1)
    #     mat.rename(columns={userId: "Scores"}, inplace=True)
    #     # mat = mat[mat[movie]>0]
    #     # mat[movie] = softmax(mat[movie])
    #     predicted_rate = (mat.values[:, 0] @ mat.values[:, 1]) / (
    #         mat.values[:, 0]
    #     ).sum()
    #     return predicted_rate

    # def predict(self, user_ratings: np.ndarray, k: int = 10):
    #     for userId in user_ratings.index[:k]:
    #         for movie in user_ratings.columns[:k]:
    #             try:
    #                 prediction = self.k_most_similar_item_user(movie, userId)
    #             except Exception:
    #                 prediction = -1
    #                 # print(prediction)
    #             user_ratings.loc[userId, movie] = prediction
    #             # print(prediction)
    #     return user_ratings

    def predict_ratings_for_user(self, user_ratings: np.ndarray, k: int = 20):
        """Returns the predicted ratings for all cats for a given set of
        user ratings by using the similarities between cat images."""
        return np.array([random.choice([0, 1]) for _ in range(len(user_ratings))])

    def recommend_k_new_items(self, user_ratings: np.ndarray, k: int = 20):
        predicted_ratings = self.predict_ratings_for_user(user_ratings, k)
        return np.argsort(predicted_ratings)[::-1][:k]

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
