import pandas as pd

from mmtd_recommender.models import ItemBasedCFRecommender, PopularityRecommender, tracks_for_user


def test_tracks_for_user_collapses_duplicates_without_dropping_them(tweets_df):
    # user 100 tweeted track 10 three times; the old keep=False logic dropped it
    # entirely instead of collapsing it to one entry.
    tracks = tracks_for_user(tweets_df, 100)
    assert sorted(tracks) == [10, 20]


def test_tracks_for_user_unknown_user_returns_empty(tweets_df):
    assert tracks_for_user(tweets_df, 999) == []


def test_popularity_top_songs_orders_by_tweet_count(tweets_df, tracks_df, artists_df):
    recommender = PopularityRecommender(tweets_df, tracks_df, artists_df)
    top = recommender.top_songs(topn=1)
    assert top.iloc[0]["track_id"] == 10
    assert top.iloc[0]["track_title"] == "Track Ten"
    assert top.iloc[0]["artist_name"] == "Artist One"


def test_popularity_top_songs_respects_topn(tweets_df, tracks_df, artists_df):
    recommender = PopularityRecommender(tweets_df, tracks_df, artists_df)
    assert len(recommender.top_songs(topn=2)) == 2


def test_item_cf_excludes_users_own_tracks(tweets_df, tracks_df, artists_df):
    recommender = ItemBasedCFRecommender(tweets_df, tracks_df, artists_df)
    user_tracks = tracks_for_user(tweets_df, 100)  # [10, 20]

    result = recommender.recommend(user_tracks, topn=10)

    assert set(result["track_id"]) & set(user_tracks) == set()


def test_item_cf_recommends_track_shared_by_similar_users(tweets_df, tracks_df, artists_df):
    recommender = ItemBasedCFRecommender(tweets_df, tracks_df, artists_df)
    user_tracks = tracks_for_user(tweets_df, 100)  # [10, 20]

    result = recommender.recommend(user_tracks, topn=10)

    # users 101 and 102 each share one track with user 100 and both also tweeted
    # track 30, so it should surface as a recommendation.
    assert 30 in result["track_id"].values


def test_item_cf_empty_user_tracks_returns_empty_frame(tweets_df, tracks_df, artists_df):
    recommender = ItemBasedCFRecommender(tweets_df, tracks_df, artists_df)
    result = recommender.recommend([], topn=10)
    assert result.empty
    assert list(result.columns) == ["track_id", "score", "track_title", "artist_name"]


def test_item_cf_unknown_track_ignored_gracefully(tweets_df, tracks_df, artists_df):
    recommender = ItemBasedCFRecommender(tweets_df, tracks_df, artists_df)
    result = recommender.recommend([99999], topn=10)
    assert result.empty
