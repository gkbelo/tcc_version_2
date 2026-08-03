"""Recommenders over the MMTD tweet/track/artist tables."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity


def tracks_for_user(tweets_df: pd.DataFrame, user_id: int) -> list[int]:
    """Distinct track ids a user has tweeted about."""
    return tweets_df.loc[tweets_df["tweet_userId"] == user_id, "tweet_trackId"].unique().tolist()


def _with_track_and_artist(scores: pd.Series, tracks_df: pd.DataFrame, artists_df: pd.DataFrame) -> pd.DataFrame:
    result = scores.reset_index()
    result.columns = ["track_id", "score"]
    result = result.merge(tracks_df, on="track_id", how="left")
    result = result.merge(artists_df, left_on="track_artistId", right_on="artist_id", how="left")
    return result.drop(columns=["artist_id", "track_artistId"])


class PopularityRecommender:
    """Recommends the tracks with the most tweets overall."""

    def __init__(self, tweets_df: pd.DataFrame, tracks_df: pd.DataFrame, artists_df: pd.DataFrame):
        self.tweets_df = tweets_df
        self.tracks_df = tracks_df
        self.artists_df = artists_df

    def top_songs(self, topn: int = 10) -> pd.DataFrame:
        counts = self.tweets_df.groupby("tweet_trackId")["tweet_count"].sum().sort_values(ascending=False)
        counts.index.name = "track_id"
        top = counts.rename("score").head(topn)
        return _with_track_and_artist(top, self.tracks_df, self.artists_df)


class ItemBasedCFRecommender:
    """Item-based collaborative filtering: recommends tracks similar to ones a user
    already tweeted about, using cosine similarity over a sparse track x user
    interaction matrix (weighted by tweet count).
    """

    def __init__(self, tweets_df: pd.DataFrame, tracks_df: pd.DataFrame, artists_df: pd.DataFrame):
        self.tweets_df = tweets_df
        self.tracks_df = tracks_df
        self.artists_df = artists_df
        self._fit()

    def _fit(self) -> None:
        interactions = (
            self.tweets_df.groupby(["tweet_trackId", "tweet_userId"])["tweet_count"].sum().reset_index()
        )

        track_codes = interactions["tweet_trackId"].astype("category")
        user_codes = interactions["tweet_userId"].astype("category")
        self._track_ids = track_codes.cat.categories.to_numpy()
        self._track_id_to_idx = {track_id: idx for idx, track_id in enumerate(self._track_ids)}

        item_user_matrix = csr_matrix(
            (interactions["tweet_count"].to_numpy(), (track_codes.cat.codes, user_codes.cat.codes)),
            shape=(len(self._track_ids), len(user_codes.cat.categories)),
        )
        self._similarity = cosine_similarity(item_user_matrix, dense_output=False)

    def recommend(self, user_tracks: list[int], ignore_tracks: list[int] | None = None, topn: int = 10) -> pd.DataFrame:
        if ignore_tracks is None:
            ignore_tracks = user_tracks

        idxs = [self._track_id_to_idx[t] for t in user_tracks if t in self._track_id_to_idx]
        if not idxs:
            return _with_track_and_artist(pd.Series(dtype=float, name="score"), self.tracks_df, self.artists_df)

        scores = np.asarray(self._similarity[idxs].sum(axis=0)).ravel()
        scores_series = pd.Series(scores, index=self._track_ids, name="score")
        scores_series.index.name = "track_id"

        known = [t for t in ignore_tracks if t in scores_series.index]
        scores_series = scores_series.drop(index=known)

        top = scores_series[scores_series > 0].sort_values(ascending=False).head(topn)
        return _with_track_and_artist(top, self.tracks_df, self.artists_df)
