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
    2695: "1 tweet",
    855: "2 tweets",
    3249: "3 tweets",
    704953: "4 tweets",
    2148401: "5 tweets",
    754714: "6 tweets",
    335: "7 tweets",
    820922: "8 tweets",
    14497584: "9 tweets",
    13498312: "10 tweets",
    9545542: "13 tweets",
    7397482: "17 tweets",
    3104781: "22 tweets",
    15134035: "29 tweets",
    18835209: "38 tweets",
    25876319: "49 tweets",
    19596311: "63 tweets",
    33063653: "82 tweets",
    273886726: "106 tweets",
    47048507: "137 tweets",
    128711456: "178 tweets",
    319862449: "229 tweets",
    302244949: "302 tweets",
    284679515: "385 tweets",
    52793396: "506 tweets",
    182436702: "664 tweets",
    244269766: "741 tweets",
    23573258: "812 tweets",
    258435110: "815 tweets",
    147606305: "939 tweets",
    102754062: "988 tweets",
    75238949: "1076 tweets",
    27458970: "1096 tweets",
    199729912: "1172 tweets",
    583487565: "1176 tweets",
    370457976: "1333 tweets",
    93769355: "2046 tweets",
    83578673: "2240 tweets",
    422939499: "6609 tweets",
    26432623: "20018 tweets",
    174194590: "32253 tweets",
    174626103: "32693 tweets",
    174228242: "35768 tweets",
    161262801: "59575 tweets",
    160874621: "89317 tweets",
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
