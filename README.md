# MMTD Song Recommender

Song recommendation models over the [MMTD (Million Musical Tweets Dataset)](https://github.com/gkbelo/tcc_code). Originally a TCC (university thesis) project written as a single Google Colab notebook; refactored into an installable package with tests.

## What it does

Two recommenders over ~1M tweets linking users to songs they listened to:

- **`PopularityRecommender`** — top tracks by total tweet count. Used as the cold-start fallback for users with no history.
- **`ItemBasedCFRecommender`** — item-based collaborative filtering. Builds a sparse track×user interaction matrix (weighted by tweet count) and recommends tracks with the highest cosine similarity to what a user has already tweeted about.

## Project layout

```
mmtd_recommender/
  data.py      # download raw dataset files, load into DataFrames
  models.py    # PopularityRecommender, ItemBasedCFRecommender
tests/         # pytest suite (synthetic fixtures, no network/data download needed)
recommend.ipynb  # thin demo notebook: load data, run both models
app.py         # Streamlit screen: pick a user_id, get recommendations
```

## Setup

```bash
python -m venv .venv
./.venv/Scripts/pip install -e ".[dev,app]"
```

## Usage

```python
from pathlib import Path
from mmtd_recommender import download_dataset, load_artists, load_tracks, load_tweets
from mmtd_recommender import PopularityRecommender, ItemBasedCFRecommender, tracks_for_user

paths = download_dataset(Path("data/raw"))
artists = load_artists(paths["artists.txt"])
tracks = load_tracks(paths["track.txt"])
tweets = load_tweets(paths["tweet.txt"])

pop_model = PopularityRecommender(tweets, tracks, artists)
pop_model.top_songs(5)

user_tracks = tracks_for_user(tweets, user_id=265101134)
cf_model = ItemBasedCFRecommender(tweets, tracks, artists)
cf_model.recommend(user_tracks, topn=10)
```

Or open `recommend.ipynb` (kernel needs the `.venv` installed above).

## Front-end

```bash
./.venv/Scripts/streamlit run app.py
```

Opens a local screen with a user_id picker (a few example users, or type any ID), a slider for the number of recommendations, and a "Recommend" button. Falls back to the popularity model for users with no tweet history. Data loading and model fitting are cached (`st.cache_resource`) so they only run once per server process.

## Tests

```bash
./.venv/Scripts/pytest
```

## Notes on the dataset

`artists.txt` has one malformed row (id 197178): the artist name contains a literal tab (`Ivar\tJohannson`) instead of a space, which breaks a plain `sep='\t'` read. `load_artists` splits each line with `maxsplit=2` so the name field absorbs the extra tab instead of the row being dropped or the parse failing.

## Changes from the original notebook

- Replaced `!curl` Colab shell calls with a `requests`-based downloader that caches files locally.
- Fixed the `artists.txt` parse error instead of the load simply crashing.
- Fixed a dedup bug (`drop_duplicates(keep=False)`) that silently dropped any track/user with more than one tweet, instead of collapsing it to one.
- Replaced the original O(tracks × users) nested-loop recommender (a per-track, per-user Python scan over the full tweets table) with a vectorized sparse-matrix cosine-similarity model — recommendations now run in tens of milliseconds instead of minutes.
- Moved logic out of notebook globals into a tested, importable package.
