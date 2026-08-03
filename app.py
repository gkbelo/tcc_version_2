"""Streamlit screen for the MMTD song recommender.

Run with: streamlit run app.py
"""

from pathlib import Path

import streamlit as st

from mmtd_recommender import (
    ItemBasedCFRecommender,
    PopularityRecommender,
    download_dataset,
    load_artists,
    load_tracks,
    load_tweets,
    tracks_for_user,
)

DATA_DIR = Path(__file__).parent / "data" / "raw"

EXAMPLE_USERS = {
    265101134: "14 tweets",
    58937384: "854 tweets",
    92235951: "75 tweets",
    250253081: "2 tweets",
    43254: "new user, no history",
}


@st.cache_resource(show_spinner="Loading dataset...")
def load_data():
    paths = download_dataset(DATA_DIR)
    artists = load_artists(paths["artists.txt"])
    tracks = load_tracks(paths["track.txt"])
    tweets = load_tweets(paths["tweet.txt"])
    return artists, tracks, tweets


@st.cache_resource(show_spinner="Fitting recommenders...")
def build_models(_artists, _tracks, _tweets):
    popularity = PopularityRecommender(_tweets, _tracks, _artists)
    collaborative = ItemBasedCFRecommender(_tweets, _tracks, _artists)
    return popularity, collaborative


st.title("MMTD Song Recommender")

artists, tracks, tweets = load_data()
pop_model, cf_model = build_models(artists, tracks, tweets)

st.caption(f"{len(tweets):,} tweets · {tweets['tweet_userId'].nunique():,} users · {len(tracks):,} tracks")

st.subheader("Pick a user")
example_label = st.selectbox(
    "Example users",
    options=list(EXAMPLE_USERS.keys()),
    format_func=lambda uid: f"{uid} ({EXAMPLE_USERS[uid]})",
)
user_id = st.number_input("Or enter any user ID", min_value=0, value=example_label, step=1)

topn = st.slider("Number of recommendations", min_value=1, max_value=20, value=10)

if st.button("Recommend", type="primary"):
    user_tracks = tracks_for_user(tweets, user_id)

    if not user_tracks:
        st.info(f"User {user_id} has no tweet history — showing popularity fallback.")
        recommendations = pop_model.top_songs(topn=topn)
    else:
        st.write(f"User {user_id} has tweeted about {len(user_tracks)} distinct tracks.")
        recommendations = cf_model.recommend(user_tracks, topn=topn)

    st.subheader("Recommendations")
    st.dataframe(recommendations, width="stretch")
