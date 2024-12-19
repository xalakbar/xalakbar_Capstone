import time
import streamlit as st
from bookscout_rs import(
    insert_user,
    check_credentials, 
    get_uid,
    get_goodbooks,
    get_hy_recommendations,
    get_existing_rating,
    save_rating,
    get_all_reviews,
    save_review,
    get_top_rated_books
)


@st.cache_data  # Cache the goodbooks data to avoid repeated calls to the database
def get_goodbooks_cached():
    return get_goodbooks()

@st.cache_data  # Cache recommendations to avoid recalculating them every time
def get_hy_recommendations_cached(username, work_id):
    return get_hy_recommendations(username, work_id)


def login():
    st.header("Log In")
    user_log = st.text_input("Username", key="login_username")
    pass_log = st.text_input("Password", key="login_password", type='password')

    if st.button("Login"):
        if user_log and pass_log:
            if check_credentials(user_log, pass_log):
                st.session_state['logged_in'] = True
                st.session_state['username'] = str(user_log)
                st.session_state['user_id'] = get_uid(user_log)
                st.success("Login successful!")
                time.sleep(2)
                st.rerun()
            else:
                st.error("Invalid username or password.")
        else:
            st.warning("Please fill in both fields.")


def signup():
    with st.expander("Sign Up", expanded=False):
        user_sig = st.text_input("Username", key="signup_username")
        pass_sig = st.text_input("Password", key="signup_password", type="password")
        confirm_pass = st.text_input("Confirm Password", key="signup_confirm_password", type="password")

        if st.button("Sign Up"):
            if user_sig and pass_sig and confirm_pass:
                if pass_sig == confirm_pass:
                    user_id = insert_user(user_sig, pass_sig)
                    if user_id:
                        st.session_state['logged_in'] = True
                        st.session_state['username'] = str(user_sig)
                        st.session_state['user_id'] = user_id
                        st.success("Sign-up sucessful!")
                        time.sleep(2)
                        st.rerun()
                    else:
                        st.error("Username already exists. Please choose a different one.")
                else:
                    st.error("Passwords do not match.")
            else:
                st.warning("Please fill in all fields.")


def homepage():
    username = st.session_state.get('username')
    goodbooks = get_goodbooks_cached()
    titles = goodbooks['title'].tolist()

    previous_selected_book = st.session_state.get('previous_selected_book', None)

    selected_book = st.selectbox("Search or select a book", titles, key="book_select")

    if selected_book != previous_selected_book:
        if 'recommendations' in st.session_state:
            del st.session_state['recommendations']
        st.session_state['recommendations_found'] = False

        if 'selected_book' in st.session_state:
            del st.session_state['selected_book']
        st.session_state['selected_book_updated'] = False

        st.session_state['previous_selected_book'] = selected_book

    if st.button("Get Recommendations"):
        work_id = goodbooks[goodbooks['title'] == selected_book]['work_id'].values[0]
        with st.spinner("Fetching recommendations..."):
            recommendations = get_hy_recommendations(username, work_id)

        if recommendations is not None and not recommendations.empty:
            st.session_state.recommendations = recommendations
            st.session_state.recommendations_found = True
        else:
            st.session_state.recommendations = None
            st.session_state.recommendations_found = False

        st.rerun()

    if 'recommendations_found' in st.session_state and st.session_state.get('recommendations_found', False):
            col1, col2, col3, col4, col5 = st.columns(5)
            recommendations = st.session_state.recommendations
            num_recommendations = len(recommendations)
            for index, row in recommendations.iterrows():
                book_work_id = row['work_id']
                book_image = row['image_url']
                book_title = row['title']
                book_author = row['author']
                book_description = row['description']

                col_index = index % 5
                with [col1, col2, col3, col4, col5][col_index]:
                    st.image(book_image, caption=book_title, use_column_width=True)
                    
                    if st.button("See Details", key=f"details_{book_work_id}"): #Button is back
                        st.session_state.selected_book = {
                            'id': book_work_id,
                            'title': book_title,
                            'author': book_author,
                            'description': book_description,
                        }

                        st.session_state.selected_book_updated = True

    # If no recommendations found, display a message
    if 'recommendations_found' in st.session_state and not st.session_state.get('recommendations_found', False):
        st.write("No recommendations yet. Click to find some!")

    if 'selected_book' in st.session_state and st.session_state.get('selected_book_updated', False):
        selected_book = st.session_state.selected_book
        work_id = selected_book['id']

        existing_rating = get_existing_rating(username, work_id)

        st.write(f"**Title:** {selected_book['title']}")
        st.write(f"**Author:** {selected_book['author']}")
        st.write(f"**Description:** {selected_book['description']}")

        st.write("Rate this book:")

        rating_key = f'rating_{work_id}'

        if rating_key not in st.session_state:
            st.session_state[rating_key] = existing_rating if existing_rating is not None else None

        rating = st.feedback("stars", key=f"rating_widget_{work_id}")

        if rating is not None and rating != st.session_state[rating_key]:
            adjusted_rating = rating + 1
            st.session_state[rating_key] = adjusted_rating
            save_rating(username, work_id, adjusted_rating)
            st.write(f"{adjusted_rating} ⭐️ rating saved.")

        if st.session_state[rating_key] is not None:
            st.write(f"You've rated this book {st.session_state[rating_key]} ⭐️")

        review_text = st.text_area("Leave a Review:", key=f"review_{work_id}")

        if st.button("Submit Review", key=f"submit_review_{work_id}"):
            if review_text:
                save_review(username, work_id, review_text)
                st.success("Review submitted successfully!")
                st.rerun()
            else:
                st.error("Please write something in the review field.")

        reviews_df = get_all_reviews(work_id)
        st.header("**Reviews:**")

        if not reviews_df.empty:
            for index, review in reviews_df.iterrows():
                formatted_date = review['review_date'].strftime('%Y-%m-%d')
                st.write(f"({formatted_date}) **{review['username']}**: {review['review_txt']}")
        else:
            st.write("No reviews yet for this book.")

    top_books = get_top_rated_books()

    st.title("Top Rated Books")
    col1, col2, col3, col4, col5 = st.columns(5)

    for index, row in top_books.iterrows():
        book_work_id = row['work_id']
        book_title = row['title']
        book_author = row['author']
        book_image = row['image_url']
        book_rating = round(row['avg_rating'], 2)

        col_index = index % 5
        with [col1, col2, col3, col4, col5][col_index]:
            st.image(book_image, caption=f"{book_title} by {book_author}", use_column_width=True)
            st.write(f"{book_rating} ⭐ Rating")

def main():
    main_logo = 'bookscout_logo.png'
    icon_logo = 'bookscout_icon.png'
    st.logo(main_logo, size='large')

    if 'logged_in' in st.session_state and st.session_state['logged_in']:
        st.logo(main_logo, size='large', icon_image=icon_logo)
        st.sidebar.title("Welcome to BookScout!")
        if 'username' in st.session_state:
            st.sidebar.header(f"Hello, {st.session_state['username']}!")

        if st.sidebar.button("Logout"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
                st.success("Log out successful.")
                time.sleep(2)
                st.rerun()

        homepage()
    else:
        login()
        signup()

if __name__ == "__main__":
    main()