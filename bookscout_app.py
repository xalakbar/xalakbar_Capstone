import streamlit as st

from bookscout_rs import(
    setup_database,
    reset_database,
    insert_books_from_df,
    insert_ratings_from_df,
    fetch_cb_data,
    fetch_cf_data,
    prepare_cf_data,
    load_svd,
    train_svd,
    load_embeddings,
    get_cf_recommendations,
    get_cb_recommendations,
    get_hy_recommendations
)

setup_database()

st.title("Book Recommendation System")