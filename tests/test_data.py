from mmtd_recommender.data import load_artists, load_tracks, load_tweets


def test_load_artists_handles_embedded_tab_in_name(tmp_path):
    # reproduces the real MMTD row (id 197178) where a name has a literal tab
    # instead of a space, which used to raise a pandas ParserError.
    raw = (
        "1\tmbid-1\tNormal Artist\n"
        "2\tmbid-2\tIvar\tJohannson\n"
        "3\tmbid-3\tAnother Artist\n"
    )
    path = tmp_path / "artists.txt"
    path.write_text(raw, encoding="utf-8")

    df = load_artists(path)

    assert list(df.columns) == ["artist_id", "artist_name"]
    assert len(df) == 3
    assert df.loc[df["artist_id"] == 2, "artist_name"].iloc[0] == "Ivar Johannson"


def test_load_tracks_parses_expected_columns(tmp_path):
    path = tmp_path / "track.txt"
    path.write_text("10\tTrack Title\t1\n", encoding="utf-8")

    df = load_tracks(path)

    assert list(df.columns) == ["track_id", "track_title", "track_artistId"]
    assert df.iloc[0]["track_id"] == 10


def test_load_tweets_drops_unused_columns_and_adds_count(tmp_path):
    path = tmp_path / "tweet.txt"
    path.write_text(
        "1\t123456\t100\t1\t10\t2012-02-09 00:43:00\t3\t0.1\t51.2\n",
        encoding="utf-8",
    )

    df = load_tweets(path)

    assert "tweet_datetime" not in df.columns
    assert "tweet_weekday" not in df.columns
    assert df.iloc[0]["tweet_count"] == 1
    assert df.iloc[0]["tweet_userId"] == 100
