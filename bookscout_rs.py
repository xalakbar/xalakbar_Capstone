import time
import pickle
import hashlib
import pandas as pd
import numpy as np
import pyodbc
import streamlit as st
import logging
from logging.handlers import RotatingFileHandler

handler = RotatingFileHandler('bookscout_rs.log', maxBytes=5 * 1024 * 1024, backupCount=3)  # 5MB limit
logging.basicConfig(
    filename='bookscout_rs.log',
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
    

# For sentiment analysis
def initialize_nltk():
    import nltk
    try:
        nltk.data.find('sentiment/vader_lexicon.zip')
    except LookupError:
        logging.warning("Vader lexicon not found, downloading...")
        nltk.download('vader_lexicon')


@st.cache_resource
def get_db_connection():
# Connection string for Azure SQL database
    connection_string = (
        "Driver={ODBC Driver 17 for SQL Server};"
        "Server=bookscoutrs-server.database.windows.net;"
        "Database=bookscoutrs;"
        "Uid=bookscoutrs_admin;"
        "Pwd=Alohabooks24;"
        "Timeout=60;"
    )

    retries = 3
    for attempt in range(retries):
            try:
                conn = pyodbc.connect(connection_string)
                logging.info(f"Database connection successful on attempt {attempt + 1}.")
                return conn
            except pyodbc.OperationalError as e:
                logging.error(f"Connection failed on attempt {attempt + 1}: {e}")
                if attempt < retries - 1:
                    time.sleep(5)
                else:
                    logging.critical("Failed to connect to the database after several attempts.")
                    raise Exception("Failed to connect to the database after several attempts.")



# Collaborative filtering (CF) data retrieval
@st.cache_data
def get_cf_data():
    try:
        with get_db_connection() as conn:
            query = '''
                SELECT r.rating, r.username, r.work_id, 
                    COALESCE(rv.sentiment_score, 0) AS sentiment_score
                FROM ratings r
                LEFT JOIN reviews rv ON r.username = rv.username AND r.work_id = rv.work_id
            '''
            ratings_df = pd.read_sql_query(query, conn)

        logging.info(f"Successfully retrieved {len(ratings_df)} ratings from the database.")
        return ratings_df
    except Exception as e:
        logging.error(f"Error retrieving collaborative filtering data: {e}")
        return pd.DataFrame() 


# Prepare data for Surprise library
def prepare_cf_data(ratings_df):
    from surprise import Dataset, Reader
    try:
        reader = Reader(rating_scale=(1, 5))
        data = Dataset.load_from_df(ratings_df[['rating', 'username', 'work_id']], reader)
        logging.info(f"Successfully prepared collaborative filtering data for {len(ratings_df)} ratings.")
        return data
    except Exception as e:
        logging.error(f"Error preparing collaborative filtering data: {e}")
        return None  # Return None if there is an error


def load_svd():
    try:
        with open('svd.pkl', 'rb') as f:
            logging.info("SVD model loaded successfully.")
            return pickle.load(f)
    except FileNotFoundError:
        logging.info("SVD model loaded successfully.")
        return None


def train_svd(data):
    from surprise import SVD
    svd = load_svd()
    if svd:
        logging.info("Using pre-trained SVD model.")
        return svd
    
    from surprise.model_selection import train_test_split
    try:
        trainset, testset = train_test_split(data, test_size=0.25)
        logging.info("SVD training dataset split successful.")
    except Exception as e:
        logging.error(f"Error during train-test split: {e}")
        raise

    try:
        svd = SVD(n_factors=310, n_epochs=130, lr_all=0.017, reg_all=0.02, random_state=42)
        svd.fit(trainset)
        logging.info("SVD model training completed.")
    except Exception as e:
        logging.error(f"Error during SVD model training: {e}")
        raise

    try:
        with open('svd.pkl', 'wb') as f:
            pickle.dump(svd, f)
        logging.info("SVD model saved successfully.")
    except Exception as e:
        logging.error(f"Error saving SVD model: {e}")
        raise

    return svd


@st.cache_data
def get_cf_recommendations(username):
    try:
        ratings_df = get_cf_data()
        data = prepare_cf_data(ratings_df)
        model = train_svd(data)
        
        all_books = ratings_df['work_id'].unique()
        rated_books = ratings_df[(ratings_df['username'] == username) & (ratings_df['rating'] > 0)]['work_id']
        predictions = []
    
        for book in all_books:
            if book not in rated_books:
                # Predict ratings using SVD model
                pred = model.predict(username, book)
                predictions.append((book, pred.est))
    
        # Sort predictions by the highest predicted rating
        predictions.sort(key=lambda x: x[1], reverse=True)
        top_cf_recommendations = predictions[:30]

        logging.info(f"CF recommendations for {username} generated successfully.")
        return top_cf_recommendations
    
    except Exception as e:
        logging.error(f"Error generating CF recommendations for {username}: {e}")
        return []


# Utility to calculate a hash of the dataset to detect changes
def calculate_data_hash(df):
    return hashlib.md5(pd.util.hash_pandas_object(df).values).hexdigest()


def load_rfr():
    try:
        with open('rfr.pkl', 'rb') as f:
            rfr = pickle.load(f)

        with open('rank_df.pkl', 'rb') as f:
            rank_df = pickle.load(f)

        # Load the dataset hash to check if it has changed
        with open('dataset_hash.pkl', 'rb') as f:
            dataset_hash = pickle.load(f)

        return rfr, rank_df, dataset_hash
    
    except FileNotFoundError:
        logging.warning("One or more files (rfr.pkl, rank_df.pkl, dataset_hash.pkl) not found.")
        return None, None, None
    

# Train the model only if the dataset has changed significantly
def train_rfr():
    from sklearn.ensemble import RandomForestRegressor
    rfr, rank_df, dataset_hash_old = load_rfr()
    
    # If the model and rank_df exist, check if the dataset has changed
    if rfr is not None and rank_df is not None:
        # Load the current dataset
        ratings_df = get_cf_data()
        books_df = get_cb_data()

        rank_df_new = ratings_df.merge(books_df, on='work_id', how='left')
        current_dataset_hash = calculate_data_hash(rank_df_new)
        
        # If dataset hasn't changed, skip retraining
        if current_dataset_hash == dataset_hash_old:
            logging.info("Dataset has not changed. Skipping retraining.")
            return rfr, rank_df
    
    try:

        # Dataset has changed or no previous model, proceed with training
        ratings_df = get_cf_data()
        books_df = get_cb_data()

        rank_df = ratings_df.merge(books_df, on='work_id', how='left')

        # Load and preprocess the embeddings
        embeddings_float = load_embeddings(books_df)
        embeddings_float = np.array(embeddings_float)

        embedding_columns = [f'embedding_float_{i}' for i in range(embeddings_float.shape[1])]

        embeddings_df = pd.DataFrame(embeddings_float, columns=embedding_columns)

        rank_df = rank_df.merge(embeddings_df, left_on='work_id', right_index=True, how='left')

        # Determine target and features
        X = rank_df[['sentiment_score'] + embedding_columns]
        y = rank_df['rating']

        rfr = RandomForestRegressor(max_depth=None, min_samples_leaf=1,
                                    min_samples_split=2, n_estimators=50,
                                    random_state=42)

        rfr.fit(X, y)
        logging.info("Random Forest Regressor model trained successfully.")

        # Save model, rank_df, and dataset hash
        with open('rfr.pkl', 'wb') as f:
            pickle.dump(rfr, f)

        with open('rank_df.pkl', 'wb') as f:
            pickle.dump(rank_df, f)
        
        # Save the new dataset hash
        current_dataset_hash = calculate_data_hash(rank_df)
        with open('dataset_hash.pkl', 'wb') as f:
            pickle.dump(current_dataset_hash, f)
        
        logging.info("Random Forest model, rank_df, and dataset hash saved successfully.")
        return rfr, rank_df

    except Exception as e:
            logging.error(f"Error during Random Forest training or saving: {e}")
            raise


# Content-based filtering (CB) data retrieval
@st.cache_data
def get_cb_data():
    try:
        with get_db_connection() as conn:
                query = '''
                    SELECT work_id, title, author, description, desc_emb, image_url 
                    FROM books
                '''
                books_df = pd.read_sql_query(query, conn)
                logging.info("CB data retrieved successfully from the database.")
        return books_df
    except Exception as e:
        logging.error(f"Error retrieving CB data from the database: {e}")
        raise


# Load book description embeddings for CB
def load_embeddings(books_df):
    embeddings_float = []
    embedding_dim = None  # Store the expected embedding dimension
    error_count = 0

    for i, embedding_bytes in enumerate(books_df['desc_emb']):
        if embedding_bytes is not None:
            try:
                emb = np.frombuffer(embedding_bytes, dtype=np.float32)

                if embedding_dim is None:  # Set the expected dimension
                    embedding_dim = emb.shape[0]
                elif emb.shape[0] != embedding_dim:  # Check for consistency
                    logging.warning(f"Inconsistent embedding length at index {i}. Expected {embedding_dim}, got {emb.shape[0]}.")
                    error_count += 1
                    continue

                embeddings_float.append(emb)
            except Exception as e:
                logging.error(f"Error converting bytes to array at index {i}: {e}")
                error_count += 1
                continue
        else:
            logging.warning(f"Null embedding at index {i}.")
            error_count += 1
            continue

    if error_count > 0:
        logging.warning(f"Total number of errors in embeddings: {error_count}")

    if not embeddings_float:
        logging.error("No valid embeddings found.")
        raise ValueError("No valid embeddings found.")

    logging.info(f"Loaded {len(embeddings_float)} valid embeddings.")
    return np.stack(embeddings_float)


@st.cache_data
def get_cb_recommendations(work_id, username):
    try:
        books_df = get_cb_data()
        all_embeddings = load_embeddings(books_df)
        
        if all_embeddings.size == 0:
            logging.error("No valid embeddings found.")
            raise ValueError("No valid embeddings found.")
        
        rfr, rank_df = train_rfr()
    
        # Initialize hnswlib index
        import hnswlib
        dim = all_embeddings.shape[1]
        num_entries = all_embeddings.shape[0]
        hnsw_index = hnswlib.Index(space='cosine', dim=dim)
        hnsw_index.init_index(max_elements=num_entries, ef_construction=300, M=32)
        hnsw_index.add_items(all_embeddings, ids=np.arange(num_entries))
        hnsw_index.set_ef(80) # Runtime search parameter 

        book_index = books_df[books_df['work_id'] == work_id].index
        if book_index.size == 0:
            raise ValueError("No valid book index found.")

        # Extract embeddings for the target book and reshape for querying
        query_embedding = all_embeddings[book_index[0]].reshape(1, -1)

        # Find top 10 nearest neighbors based on cosine similarity 
        labels, distances = hnsw_index.knn_query(query_embedding, k=10)

        # Convert distances to similarity scores (1 - cosine distance)
        similarities = 1 - distances[0]

        # Retrieve recommended books using valid labels
        valid_labels = [label for label in labels[0] if label < len(books_df)]
        if not valid_labels:
            logging.warning("No valid recommendations found for work_id: {work_id}.")
            return []

        top_cb_recommendations = books_df.iloc[valid_labels][['work_id', 'title']].copy()
        top_cb_recommendations['similarity_score'] = similarities[:len(valid_labels)]
        
        # Drop duplicates and exclude selected book from recommendations 
        top_cb_recommendations = top_cb_recommendations[top_cb_recommendations['work_id'] != work_id]

        # Sort recommendations by similarity score
        top_cb_recommendations = top_cb_recommendations.sort_values(by='similarity_score', ascending=False)[:30]

        # Prepare RFR predictions (rating-based)
        all_books = books_df['work_id'].unique()
        rated_books = get_cf_data()[(get_cf_data()['username'] == username) & (get_cf_data()['rating'] > 0)]['work_id']
        rfr_predictions = []

    # Use pre-calculated embeddings and lookup by index
        for i, book in enumerate(all_books): #Use the index here
            if book not in rated_books and book != work_id:
                #Use index to lookup the correct embedding from all_embeddings
                book_embedding = all_embeddings[i].reshape(1,-1)

                sentiment_score = rank_df.loc[rank_df['work_id'] == book, 'sentiment_score'].iloc[0] if (rank_df['work_id'] == book).any() else 0

                if np.isnan(sentiment_score) or np.isinf(sentiment_score) or np.any(np.isnan(book_embedding)) or np.any(np.isinf(book_embedding)):
                    logging.warning(f"Skipping book {book} due to NaN or Inf values.")
                    continue

                book_features = np.concatenate(([sentiment_score], book_embedding[0]))
                try:
                    rfr_pred = rfr.predict(book_features.reshape(1, -1))[0]
                    rfr_predictions.append((book, rfr_pred))
                except Exception as e:
                    logging.error(f"Error predicting rating for book {book}: {e}")
                    continue

    # Create DataFrame and drop duplicates before merging
        top_rfr_recommendations_df = pd.DataFrame(rfr_predictions, columns=['work_id', 'predicted_rating'])
        top_rfr_recommendations_df.drop_duplicates(subset='work_id', inplace=True)

        # Rename 'title' in top_cb_recommendations before merging
        top_cb_recommendations.rename(columns={'title': 'title_cb', 'work_id':'work_id_cb'}, inplace=True)

    # Merge the top RFR and top CB recommendations
        top_recommendations_df = pd.concat([
            top_cb_recommendations,
            top_rfr_recommendations_df
        ], axis=1)
        top_recommendations_df = top_recommendations_df.loc[:,~top_recommendations_df.columns.duplicated()].copy()
        top_recommendations_df['work_id'] = top_recommendations_df['work_id_cb'].fillna(top_recommendations_df['work_id'])
        top_recommendations_df = top_recommendations_df.drop(['work_id_cb'], axis=1)


        # Final ranking and merging with book details
        top_recommendations_df['final_score'] = top_recommendations_df['similarity_score'] * 0.5 + top_recommendations_df['predicted_rating'] * 0.5
        final_cb_recommendations = top_recommendations_df.sort_values(by='final_score', ascending=False).head(10)
        final_cb_recommendations = final_cb_recommendations.merge(books_df[['work_id', 'title', 'author', 'description', 'image_url']], on='work_id', how='left')

        logging.info(f"Final CB recommendations for work_id {work_id} generated successfully.")
        return final_cb_recommendations[['work_id', 'title', 'author', 'description', 'image_url', 'similarity_score','predicted_rating','final_score']]

    except Exception as e:
        logging.error(f"Error generating CB recommendations for work_id {work_id} and username {username}: {e}")
        return []


# Hybrid recommendations
@st.cache_data
def get_hy_recommendations(username, work_id):
    try:
        work_id = int(work_id)
        # Collaborative filtering recommendations
        cf_recommendations = get_cf_recommendations(username)
        cf_df = pd.DataFrame(cf_recommendations, columns=['work_id', 'predicted_rating'])
        
        # Content-based filtering recommendations 
        cb_recommendations = get_cb_recommendations(work_id, username)
        cb_df = cb_recommendations.copy()
        cb_df['predicted_rating'] = cb_df.get('similarity_score', 1.0)

        # Existing rating and sentiment
        existing_rating = get_existing_rating(username, work_id)
        existing_sentiment = get_existing_sentiment(username, work_id)

        # Set initial weights (neutral)
        cf_weight = 0.5
        cb_weight = 0.5

        # Adjust weights based on rating
        if existing_rating is not None:
            if existing_rating >= 4:
                if work_id in cf_df['work_id'].values:
                    cf_weight += 0.05
                else:
                    cb_weight += 0.05
            else:
                if work_id in cf_df['work_id'].values:
                    cf_weight -= 0.05
                else:
                    cb_weight -= 0.05

        # Adjust weights based on sentiment    
        if existing_sentiment is not None:
            if existing_sentiment >= 0.5:
                cf_weight += 0.1
            elif existing_sentiment <= -0.5:
                cb_weight += 0.1

        # Ensure non-negative weights
        cf_weight = max(cf_weight, 0)
        cb_weight = max(cb_weight, 0)

        # Normalizing weights
        total_weight = cf_weight + cb_weight
        cf_weight /= total_weight
        cb_weight /= total_weight

        # Remove overlap between CF and CB recommendations
        cfcb_books = set(cf_df['work_id']).intersection(cb_df['work_id'])
        cb_df = cb_df[~cb_df['work_id'].isin(cfcb_books)]

        # Combined CF and CB recommendatons
        combined_recommendations = pd.concat([cf_df.rename(columns={'predicted_rating': 'cf_rating'}),
                                            cb_df.rename(columns={'predicted_rating': 'cb_rating'})],
                                            ignore_index=True)

        # Aggregate ratings
        combined_recommendations = combined_recommendations.groupby('work_id').agg({'cf_rating': 'sum', 'cb_rating': 'sum'}).reset_index()
    
    # Calculate final rating using weighted avg of CF and CB ratings
        combined_recommendations['final_rating'] = (combined_recommendations['cf_rating'] * cf_weight) + (combined_recommendations['cb_rating'] * cb_weight)
        
        # Add random factor to introduce slight variation
        combined_recommendations['final_rating'] += np.random.uniform(0, 0.1, size=len(combined_recommendations))

        # Sort recommendations by final rating 
        top_hy_recommendations = combined_recommendations.sort_values(by='final_rating', ascending=False).head(5)

        # Merge recommendations with book details for the final output
        books_df = get_cb_data()
        final_recommendations = top_hy_recommendations.merge(books_df[['work_id', 'title', 'author', 'description', 'image_url']], on='work_id', how='left')
        
        logging.info(f"Hybrid recommendations for user {username} and work_id {work_id} generated successfully.")
        return final_recommendations[['work_id', 'title', 'author', 'description', 'image_url']]

    except Exception as e:
        logging.error(f"Error generating hybrid recommendations for user {username} and work_id {work_id}: {e}")
        return []


def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()


@st.cache_data
def get_uid(username):
    try:
        with get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT user_id FROM users WHERE LOWER(username) = LOWER(?)", (username,))
                user_id = cursor.fetchone()

        if user_id:
            logging.info(f"User ID retrieved for username: {username}.")
            return user_id[0]
        else:
            logging.warning(f"No user ID found for username: {username}.")
            return None
    except Exception as e:
        logging.error(f"Error retrieving user ID for username {username}: {e}")
        return None


def insert_user(username, password):
    try:
        username = username.lower()
        password_hash = hash_password(password)
        
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT username FROM users WHERE username = ?", (username,))
            existing_user = cursor.fetchone()

            if existing_user:
                logging.warning(f"User '{username}' already exists.")
                return False

            cursor.execute("INSERT INTO users (username, pass_hash) VALUES (?, ?)", (username, password_hash))
            cursor.execute("SELECT SCOPE_IDENTITY()")
            user_id = cursor.fetchone()[0]
            conn.commit()

            logging.info(f"New user '{username}' inserted successfully with user_id {user_id}.")
            return user_id
    except Exception as e:
        logging.error(f"Error inserting new user '{username}': {e}")
        return False


def check_credentials(username, password):
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM users WHERE LOWER(username) = LOWER(?) AND pass_hash=?", (username, hash_password(password)))
            user = cursor.fetchone()
        
        if user:
            logging.info(f"Credentials checked for user '{username}' - valid.")
            return user
        else:
            logging.warning(f"Invalid credentials for user '{username}'.")
            return None
    except Exception as e:
        logging.error(f"Error checking credentials for user '{username}': {e}")
        return None

    
@st.cache_data
def get_goodbooks(limit):
    try:
        with get_db_connection() as conn:
            query = f'SELECT TOP {limit} work_id, title, author, description, genres, image_url FROM books;'
            goodbooks = pd.read_sql(query, conn)
        
        goodbooks['genres'] = goodbooks['genres'].fillna('')
        goodbooks['genres'] = goodbooks['genres'].astype(str)
        
        logging.info(f"Goodbooks retrieved with limit {limit}.")
        return goodbooks
    
    except Exception as e:
        logging.error(f"Error retrieving goodbooks with limit {limit}: {e}")
        return pd.DataFrame()


@st.cache_data
def get_existing_rating(username, work_id):
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT rating FROM ratings WHERE username = ? AND work_id = ?', (username, work_id))
            existing_rating = cursor.fetchone()
        
        if existing_rating:
            logging.info(f"Existing rating found for username '{username}' and work_id {work_id}.")
            return existing_rating[0]
        else:
            logging.info(f"No existing rating found for username '{username}' and work_id {work_id}.")
            return None
    except Exception as e:
        logging.error(f"Error retrieving existing rating for user '{username}' and work_id {work_id}: {e}")
        return None


@st.cache_data
def get_existing_sentiment(username, work_id):
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT sentiment_score FROM reviews WHERE username = ? AND work_id = ?', (username, work_id))
            sentiment = cursor.fetchone()
        
        if sentiment:
            logging.info(f"Existing sentiment found for username '{username}' and work_id {work_id}.")
            return sentiment[0]
        else:
            logging.info(f"No existing sentiment found for username '{username}' and work_id {work_id}.")
            return None
    except Exception as e:
        logging.error(f"Error retrieving existing sentiment for user '{username}' and work_id {work_id}: {e}")
        return None


@st.cache_data
def get_user_review(username, work_id):
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT review_txt FROM reviews WHERE username = ? AND work_id = ?', (username, work_id))
            existing_review = cursor.fetchone()
        
        if existing_review:
            logging.info(f"Existing review found for user '{username}' and work_id {work_id}.")
            return existing_review[0]
        else:
            logging.info(f"No existing review found for user '{username}' and work_id {work_id}.")
            return None
    except Exception as e:
        logging.error(f"Error retrieving existing review for user '{username}' and work_id {work_id}: {e}")
        return None



def save_rating(rating, username, work_id):
    try:
        rating = float(rating)
        work_id = int(work_id)
        
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT user_id FROM users WHERE username = ?", (username,))
            user_id_result = cursor.fetchone()

            if user_id_result is None:
                logging.warning(f"User '{username}' not found.")
                raise ValueError(f"Username '{username}' not found.")
            
            user_id = user_id_result[0]
            
            cursor.execute('INSERT INTO ratings (rating, username, work_id, user_id) VALUES (?, ?, ?, ?)', (rating, username, work_id, user_id))
            conn.commit()

            logging.info(f"Rating of {rating} for user '{username}' and work_id {work_id} saved successfully.")
    except Exception as e:
        logging.error(f"Error saving rating for user '{username}' and work_id {work_id}: {e}")



def save_review(username, work_id, review_text):
    try:
        sentiment_score = analyze_sentiment_vader(review_text)

        with get_db_connection() as conn:
            cursor = conn.cursor()

            if review_text.strip():
                review_date = pd.to_datetime("now", utc=True).strftime('%Y-%m-%d')

                cursor.execute("SELECT 1 FROM reviews WHERE username = ? AND work_id = ?", (username, work_id))
                existing_review = cursor.fetchone()

                if existing_review:
                    cursor.execute("""
                        UPDATE reviews
                        SET review_txt = ?, review_date = ?, sentiment_score = ?
                        WHERE username = ? AND work_id = ?
                    """, (review_text, review_date, sentiment_score, username, work_id))
                    logging.info(f"Updated review for user '{username}' and work_id {work_id}.")
                else:
                    cursor.execute("""
                        INSERT INTO reviews (username, work_id, review_txt, review_date, sentiment_score)
                        VALUES (?, ?, ?, ?, ?)
                    """, (username, work_id, review_text, review_date, sentiment_score))
                    logging.info(f"Inserted new review for user '{username}' and work_id {work_id}.")

                conn.commit()
                return True
            else:
                logging.warning(f"Review text is empty for user '{username}' and work_id {work_id}.")
                return False
    except pyodbc.Error as ex:
        logging.error(f"Database error while saving review for user '{username}' and work_id {work_id}: {ex}")
        return False
    except Exception as e:
        logging.error(f"Unexpected error while saving review for user '{username}' and work_id {work_id}: {e}")
        return False


@st.cache_data   
def get_all_reviews(work_id):
    try:
        with get_db_connection() as conn:
            query = '''
                SELECT r.review_txt, u.username, r.review_date 
                FROM reviews r 
                JOIN users u ON r.username = u.username
                WHERE r.work_id = ? 
                ORDER BY r.review_date DESC
            '''
            reviews_df = pd.read_sql_query(query, conn, params=(work_id,))
        
        logging.info(f"Retrieved {len(reviews_df)} reviews for work_id {work_id}.")
        return reviews_df
    except Exception as e:
        logging.error(f"Error retrieving reviews for work_id {work_id}: {e}")
        return pd.DataFrame()


@st.cache_data
def get_top_rated_books_from_db():
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT TOP 10 b.work_id, b.title, b.author, b.image_url, AVG(r.rating) as avg_rating
                FROM books b
                JOIN ratings r ON b.work_id = r.work_id
                GROUP BY b.work_id, b.title, b.author, b.image_url
                ORDER BY avg_rating DESC
            ''')

            rows = cursor.fetchall()
            column_names = [column[0] for column in cursor.description]

            # Convert rows to a list of dictionaries
            data = []
            for row in rows:
                data.append(dict(zip(column_names, row)))

            top_rated_books = pd.DataFrame(data)
        
        logging.info(f"Retrieved top-rated books from database.")
        return top_rated_books
    except Exception as e:
        logging.error(f"Error retrieving top-rated books: {e}")
        return pd.DataFrame()


def get_top_rated_books():
    return get_top_rated_books_from_db()


def analyze_sentiment_vader(review_text):
    from nltk.sentiment import SentimentIntensityAnalyzer
    try:
        sia = SentimentIntensityAnalyzer()
        if not review_text.strip():
            return 0  # Neutral sentiment for empty reviews
        sentiment = sia.polarity_scores(review_text)
        logging.info(f"Sentiment score calculated for review: {review_text[:30]}...")  # Log first 30 characters of review text
        return sentiment['compound']
    except Exception as e:
        logging.error(f"Error analyzing sentiment for review: {review_text[:30]}... - {e}")
        return 0


@st.cache_data
def get_user_rating_count(username):
    try:
        with get_db_connection() as conn:
            query = '''
                SELECT COUNT(DISTINCT work_id) AS rated_books_count
                FROM ratings
                WHERE username = ?
            '''
            result = pd.read_sql_query(query, conn, params=(username,))
        
        rated_books_count = result['rated_books_count'].iloc[0]
        logging.info(f"User '{username}' has rated {rated_books_count} books.")
        return rated_books_count
    except Exception as e:
        logging.error(f"Error retrieving rating count for user '{username}': {e}")
        return 0
    

@st.cache_data
def get_user_review_count(username):
    try:
        with get_db_connection() as conn:
            query = '''
                SELECT COUNT(DISTINCT work_id) AS review_count
                FROM reviews
                WHERE username = ?
            '''
            result = pd.read_sql(query, conn, params=(username,))
        
        review_count = result['review_count'].iloc[0]
        logging.info(f"User '{username}' has written {review_count} reviews.")
        return review_count
    except Exception as e:
        logging.error(f"Error retrieving review count for user '{username}': {e}")
        return 0

