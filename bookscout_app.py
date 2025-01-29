import time
import streamlit as st
import pandas as pd
import plotly.express as px
import bookscout_rs


def login():
    st.header("Log In")
    user_log = st.text_input("Username", key="login_username")
    pass_log = st.text_input("Password", key="login_password", type='password')

    if st.button("Login"):
        if user_log and pass_log:
            if bookscout_rs.check_credentials(user_log, pass_log):
                st.session_state['logged_in'] = True
                st.session_state['username'] = str(user_log)
                st.session_state['user_id'] = bookscout_rs.get_uid(user_log)
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
                    user_id = bookscout_rs.insert_user(user_sig, pass_sig)
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
    goodbooks = bookscout_rs.get_goodbooks(limit=100)
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
            recommendations = bookscout_rs.get_hy_recommendations(username, work_id)

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
                    
                    if st.button("See Details", key=f"details_{book_work_id}"):
                        st.session_state.selected_book = {
                            'id': book_work_id,
                            'title': book_title,
                            'author': book_author,
                            'description': book_description,
                        }

                        st.session_state.selected_book_updated = True


    if 'recommendations_found' in st.session_state and not st.session_state.get('recommendations_found', False):
        st.write("No recommendations yet. Click to find some!")

    if 'selected_book' in st.session_state and st.session_state.get('selected_book_updated', False):
        selected_book = st.session_state.selected_book
        work_id = selected_book['id']

        existing_rating = bookscout_rs.get_existing_rating(username, work_id)

        st.write(f"**Title:** {selected_book['title']}")
        st.write(f"**Author:** {selected_book['author']}")
        st.write(f"**Description:** {selected_book['description']}")

        # Ratings
        st.write("Rate this book:")

        rating_key = f'rating_{work_id}'

        if rating_key not in st.session_state:
            st.session_state[rating_key] = existing_rating if existing_rating is not None else None

        rating = st.feedback("stars", key=f"rating_widget_{work_id}")

        if rating is not None and rating != st.session_state[rating_key]:
            adjusted_rating = rating + 1
            st.session_state[rating_key] = adjusted_rating
            bookscout_rs.save_rating(adjusted_rating, username, work_id)
            st.write(f"{adjusted_rating} ⭐️ rating saved.")

        if st.session_state[rating_key] is not None:
            st.write(f"You've rated this book {st.session_state[rating_key]} ⭐️")

        # Reviews
        review_text = st.text_area("Leave a Review:", key=f"review_{work_id}")

        if st.button("Submit Review", key=f"submit_review_{work_id}"):
            if review_text:
                bookscout_rs.save_review(username, work_id, review_text)
                st.success("Review submitted successfully!")
                st.rerun()
            else:
                st.error("Please write something in the review field.")

        reviews_df = bookscout_rs.get_all_reviews(work_id)
        st.header("**Reviews:**")

        if not reviews_df.empty:
            for index, review in reviews_df.iterrows():
                formatted_date = review['review_date'].strftime('%Y-%m-%d')
                st.write(f"({formatted_date}) **{review['username']}**: {review['review_txt']}")
        else:
            st.write("No reviews yet for this book.")

    # Top Rated Books
    top_books = bookscout_rs.get_top_rated_books()

    st.header("Top Rated Books")
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
    
    # Genre Popularity Analysis
    if 'genres' in goodbooks.columns:
        # Split the genres and handle cases where genres are empty
        goodbooks['genres'] = goodbooks['genres'].str.split(',')
        goodbooks = goodbooks.explode('genres')  # Flatten the genres into individual rows
        goodbooks['genres'] = goodbooks['genres'].str.strip()  # Remove leading/trailing spaces
        goodbooks = goodbooks[goodbooks['genres'] != '']  # Remove rows with empty genres

        if not goodbooks.empty:
            genre_counts = goodbooks['genres'].value_counts().reset_index(name='Count')
            genre_counts.columns = ['genres', 'Count']  # Rename columns for clarity

            # Plot the genre popularity
            st.header('Genre Popularity')
            fig = px.bar(genre_counts, x='genres', y='Count')
            st.plotly_chart(fig)
        else:
            st.write("No genres available.")
    else:
        st.write("No genre information available.")
        st.write(f"Available columns: {goodbooks.columns}") 

    # Author Rating Analysis (Top 10 Authors based on Average Rating)
    ratings_df = bookscout_rs.get_cf_data()  
    # Add a selectbox for the user to choose between "Average Rating" or "Book Count"
    sort_by = st.selectbox(
        "Sort authors by:",
        ["Average Rating", "Book Count"]
    )

    if 'author' in goodbooks.columns:
        # Merge goodbooks with ratings data on 'work_id'
        merged_data = pd.merge(goodbooks, ratings_df, on='work_id', how='inner')

        if sort_by == "Average Rating":
            # Group by author and calculate the average rating
            author_avg_ratings = merged_data.groupby('author')['rating'].mean().reset_index(name='avg_rating')

            # Sort by average rating (in descending order)
            top_authors = author_avg_ratings.sort_values(by='avg_rating', ascending=False).head(10)

            # Plot the top 10 authors by average rating
            st.header('Top 10 Authors by Average Rating')
            fig = px.bar(top_authors, x='author', y='avg_rating')
            st.plotly_chart(fig)

        elif sort_by == "Book Count":
            # Group by author and count the number of books for each author
            author_book_count = merged_data.groupby('author')['work_id'].count().reset_index(name='book_count')

            # Sort by book count (in descending order)
            top_authors_by_count = author_book_count.sort_values(by='book_count', ascending=False).head(10)

            # Plot the top 10 authors by book count
            st.header('Top 10 Authors by Book Count')
            fig = px.bar(top_authors_by_count, x='author', y='book_count')
            st.plotly_chart(fig)

    else:
        st.write("No author information available.")
        st.write(f"Available columns: {goodbooks.columns}")


def main():
    main_logo = 'bookscout_logo.png'
    icon_logo = 'bookscout_icon.png'
    st.logo(main_logo, size='large')

    if 'logged_in' in st.session_state and st.session_state['logged_in']:
        st.logo(main_logo, size='large', icon_image=icon_logo)

        if 'username' in st.session_state:
            st.sidebar.title(f"Hello, {st.session_state['username']}!:wave:")
            
            st.sidebar.markdown(
                """
                <style>
                    .center-text {
                        text-align: center;
                    }
                </style>
                <div class="center-text">
                    <em><br>Discover your next great read,<br> one click at a time...<br><br><br></em>
                </div>
                """, unsafe_allow_html=True
            )
       
            username = st.session_state['username']
            rated_books_count = bookscout_rs.get_user_rating_count(username)
            review_count = bookscout_rs.get_user_review_count(username)

            st.sidebar.header("User Activity", divider='grey')
            # Display the counts in the sidebar
            st.sidebar.write(f"- You've rated {rated_books_count} books.:books:")
            st.sidebar.write(f"- You've left {review_count} reviews.:writing_hand:")

            data = {
                'Category': ['Books Rated', 'Reviews Left'],
                'Count': [rated_books_count, review_count]
            }
            df = pd.DataFrame(data)

            # Plot the bar chart
            fig = px.bar(df, x='Category', y='Count', color='Category')
            st.sidebar.plotly_chart(fig, use_container_width=True)

        st.sidebar.markdown('<div class="spacer"><br></div>', unsafe_allow_html=True)
        st.sidebar.markdown(
            """
            ## About BookScout 📚
            **BookScout** is your personalized book discovery tool designed to help you find your next great read. 
            
            Whether you're an avid reader or just getting started, BookScout offers book recommendations based on ratings, reviews, and popular genres to make your search easier and more enjoyable.

            ### Features:
            - **Find Top Rated Books**: Browse books based on user ratings and reviews.
            - **Track Your Reading Activity**: Keep track of your rated books and reviews.
            - **Personalized Recommendations**: Get suggestions based on your reading habits.

            ### Contact:
            Have feedback or suggestions? Reach us at:
            - Email: [support@bookscout.com](mailto:support@bookscout.com)
            - Twitter: [@BookScoutApp](https://twitter.com/BookScoutApp)
            - Instagram: [@BookScout](https://instagram.com/BookScout)

            ### How It Works:
            1. **Sign Up**: Create an account and start rating books.
            2. **Get Recommendations**: Based on your ratings, receive personalized book suggestions.
            3. **Track Progress**: View your reading stats, books rated, and reviews left.
            """, unsafe_allow_html=True)
        
        st.sidebar.markdown('<div class="spacer"><br><br><br></div>', unsafe_allow_html=True)
        # Logout Functionality
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