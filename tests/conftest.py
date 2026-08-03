import pandas as pd
import pytest


@pytest.fixture
def artists_df():
    return pd.DataFrame(
        {
            "artist_id": [1, 2, 3],
            "artist_name": ["Artist One", "Artist Two", "Artist Three"],
        }
    )


@pytest.fixture
def tracks_df():
    return pd.DataFrame(
        {
            "track_id": [10, 20, 30],
            "track_title": ["Track Ten", "Track Twenty", "Track Thirty"],
            "track_artistId": [1, 2, 3],
        }
    )


@pytest.fixture
def tweets_df():
    # user 100 tweeted track 10 three times (to exercise dedup) and track 20 once
    # user 101 tweeted track 10 and track 30
    # user 102 tweeted track 20 and track 30
    rows = [
        (1, 100, 10),
        (2, 100, 10),
        (3, 100, 10),
        (4, 100, 20),
        (5, 101, 10),
        (6, 101, 30),
        (7, 102, 20),
        (8, 102, 30),
    ]
    df = pd.DataFrame(rows, columns=["tweet_id", "tweet_userId", "tweet_trackId"])
    df["tweet_tweetId"] = df["tweet_id"] + 1000
    df["tweet_artistId"] = 0
    df["tweet_count"] = 1
    return df
