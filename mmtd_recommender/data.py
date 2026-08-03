"""Download and load the MMTD (Million Musical Tweets Dataset) text files."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import requests

logger = logging.getLogger(__name__)

BASE_URL = "https://raw.githubusercontent.com/gkbelo/tcc_code/master/mmtd"
FILES = ("artists.txt", "track.txt", "tweet.txt")


def download_dataset(dest_dir: Path, force: bool = False) -> dict[str, Path]:
    """Download the MMTD raw files into dest_dir, skipping files that already exist."""
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    paths = {}
    for filename in FILES:
        dest = dest_dir / filename
        paths[filename] = dest
        if dest.exists() and not force:
            logger.info("skipping %s (already downloaded)", filename)
            continue

        url = f"{BASE_URL}/{filename}"
        logger.info("downloading %s", url)
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        dest.write_bytes(response.content)

    return paths


def load_artists(path: Path) -> pd.DataFrame:
    """Load artists.txt as (artist_id, artist_name).

    A handful of artist names contain a literal tab character (e.g. "Ivar\tJohannson"),
    which breaks a plain sep='\\t' read (pandas sees 4 fields instead of 3 on that row).
    Splitting each line with maxsplit=2 keeps the id/mbid columns intact and lets any
    extra tabs fall into the name field, where they're normalized to a single space.
    """
    ids, names = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t", 2)
            artist_id, _mbid, name = parts
            ids.append(int(artist_id))
            names.append(" ".join(name.split("\t")))

    return pd.DataFrame({"artist_id": ids, "artist_name": names})


def load_tracks(path: Path) -> pd.DataFrame:
    cols = ["track_id", "track_title", "track_artistId"]
    return pd.read_csv(path, sep="\t", names=cols)


def load_tweets(path: Path) -> pd.DataFrame:
    cols = [
        "tweet_id",
        "tweet_tweetId",
        "tweet_userId",
        "tweet_artistId",
        "tweet_trackId",
        "tweet_datetime",
        "tweet_weekday",
        "tweet_longitude",
        "tweet_latitude",
    ]
    df = pd.read_csv(path, sep="\t", names=cols)
    df = df.drop(columns=["tweet_weekday", "tweet_longitude", "tweet_latitude", "tweet_datetime"])
    df["tweet_count"] = 1
    return df
