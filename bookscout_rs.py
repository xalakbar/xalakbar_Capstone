import os
import pickle
import sqlite3
import hashlib
import pandas as pd
import numpy as np
import hnswlib
import nltk
import pyodbc
import streamlit as st
from nltk.sentiment import SentimentIntensityAnalyzer
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# BEGIN SQLITE3
# Initially used SQLite3 for db; Moved over to Azure SQL database. 
def setup_database():
    conn = sqlite3.connect('bookscout.db')
    cursor = conn.cursor()
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS books (
        work_id INTEGER PRIMARY KEY,
        title TEXT,
        author TEXT,
        description TEXT,
        genres TEXT,
        desc_emb BLOB,
        image_url TEXT
    );
    ''')
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        user_id INTEGER PRIMARY KEY,
        username TEXT NOT NULL,
        pass_hash TEXT NOT NULL
    );
    ''')
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS ratings (
        rating INTEGER,
        username TEXT NOT NULL,
        work_id INTEGER,
        UNIQUE(username, work_id),
        FOREIGN KEY (username) REFERENCES users(username),
        FOREIGN KEY (work_id) REFERENCES books(work_id)
    );
    ''')
        
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS reviews (
        username INTEGER,
        work_id INTEGER,
        review_txt TEXT,
        review_date DATETIME,
        sentiment_score REAL,
        FOREIGN KEY (username) REFERENCES users(username),
        FOREIGN KEY (work_id) REFERENCES books(work_id)
    );
    ''')
    
    conn.commit()
    conn.close()


def insert_books_from_df(df):
    conn = sqlite3.connect('bookscout.db')
    cursor = conn.cursor()
    for _, row in df.iterrows():
        # Convert embeddings to bytes for SQLite compatability
        embeddings = np.array(row['scaled_desc_emb']).tobytes() if 'scaled_desc_emb' in row else None
        cursor.execute('''
        INSERT OR IGNORE INTO books (work_id, title, author, description, genres, desc_emb, image_url) VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (row['work_id'], row['title'], row['authors'], row['description'], row['genres'], embeddings, row['image_url']))
    conn.commit()
    conn.close()


def insert_users_from_df(df):
    conn = sqlite3.connect('bookscout.db')
    cursor = conn.cursor()
    
    uid_mapping = {} # Dictionary to store username-to-user_id mapping

    for _, row in df.iterrows():
        username = row['username']
        pass_hash = row['password_hash']

        cursor.execute('SELECT user_id FROM users WHERE LOWER(username) = LOWER(?)', (username.lower(),))
        existing_user = cursor.fetchone()

        if existing_user:
            user_id = existing_user[0]
        else:
            try:
               cursor.execute('''
                    INSERT INTO users (username, pass_hash) VALUES (?, ?)
                              ''', (username.lower(), pass_hash))
               user_id = cursor.lastrowid
            except sqlite3.IntegrityError:
                continue

        uid_mapping[username.lower()] = user_id # Map username to user_id

    conn.commit()
    conn.close()
    return uid_mapping


def insert_ratings_from_df(df, uid_mapping):
    conn = sqlite3.connect('bookscout.db')
    cursor = conn.cursor()

    for _, row in df.iterrows():
        username = row['username']

        if username is not None:
            work_id = row['work_id']
            rating = row['rating']
            
            cursor.execute('''
            INSERT OR REPLACE INTO ratings (rating, username, work_id) VALUES (?, ?, ?)
            ''', (rating, username, work_id))

    conn.commit()
    conn.close()


def reset_database():
    if os.path.exists('bookscout.db'):
        os.remove('bookscout.db')

# END SQLITE3


# For sentiment analysis
def initialize_nltk():
    try:
        nltk.data.find('sentiment/vader_lexicon.zip')
    except LookupError:
        nltk.download('vader_lexicon')


# Connection string for Azure SQL database
connection_string = (
    "Driver={ODBC Driver 17 for SQL Server};"
    "Server=bookscoutrs-server.database.windows.net;"
    "Database=bookscoutrs;"
    "Uid=bookscoutrs_admin;"
    "Pwd=Alohabooks24;"
    "Timeout=60;"
)


@st.cache_resource
def get_db_connection():
    conn = pyodbc.connect(connection_string)
    return conn


# Collaborative filtering (CF) data retrieval
@st.cache_data
def get_cf_data():
    conn = get_db_connection()
    query = '''
        SELECT r.rating, r.username, r.work_id, 
               COALESCE(rv.sentiment_score, 0) AS sentiment_score
        FROM ratings r
        LEFT JOIN reviews rv ON r.username = rv.username AND r.work_id = rv.work_id
    '''
    ratings_df = pd.read_sql_query(query, conn)
    conn.close()

    return ratings_df


# Prepare data for Surprise library
def prepare_cf_data(ratings_df):
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(ratings_df[['rating', 'username', 'work_id']], reader)

    return data


def load_svd():
    try:
        with open('svd.pkl', 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None

@st.cache_data
def train_svd(data):
    svd = load_svd()
    if svd:
        return svd
    
    trainset, testset = train_test_split(data, test_size=0.25)
    svd = SVD(n_factors=310, n_epochs=130, lr_all=0.017, reg_all=0.02, random_state=42)
    svd.fit(trainset)
    
    with open('svd.pkl', 'wb') as f:
        pickle.dump(svd, f)

    return svd

@st.cache_data
def get_cf_recommendations(username):
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

    return top_cf_recommendations


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
        return None, None, None
    

# Train the model only if the dataset has changed significantly
@st.cache_data
def train_rfr():
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
            return rfr, rank_df
        
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

    # Save model, rank_df, and dataset hash
    with open('rfr.pkl', 'wb') as f:
        pickle.dump(rfr, f)

    with open('rank_df.pkl', 'wb') as f:
        pickle.dump(rank_df, f)
    
    # Save the new dataset hash
    current_dataset_hash = calculate_data_hash(rank_df)
    with open('dataset_hash.pkl', 'wb') as f:
        pickle.dump(current_dataset_hash, f)
    
    return rfr, rank_df


# Content-based filtering (CB) data retrieval
@st.cache_data
def get_cb_data():
    conn = get_db_connection()
    books_df = pd.read_sql_query('SELECT work_id, title, author, description, desc_emb, image_url FROM books', conn)
    conn.close()

    return books_df


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
                    print(f"Error: Inconsistent embedding length at index {i}. Expected {embedding_dim}, got {emb.shape[0]}.")
                    error_count += 1
                    continue  # Skip this embedding

                embeddings_float.append(emb)
            except Exception as e:
                print(f"Error converting bytes to array at index {i}: {e}")
                error_count += 1
                continue
        else:
            print(f"Error: Null embedding at index {i}.")
            error_count += 1
            continue

    if error_count > 0:
        print(f"Total number of errors: {error_count}")

    if not embeddings_float:
        raise ValueError("No valid embeddings found.")

    return np.stack(embeddings_float)


@st.cache_data
def get_cb_recommendations(work_id, username):
    books_df = get_cb_data()
    all_embeddings = load_embeddings(books_df)
    
    if all_embeddings.size == 0:
        raise ValueError("No valid embeddings found.")
    
    rfr, rank_df = train_rfr()
    
    # Initialize hnswlib index
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
        print("No valid recommendations found.")
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
                print(f"Skipping book {book} due to NaN or Inf values.")
                continue

            book_features = np.concatenate(([sentiment_score], book_embedding[0]))
            rfr_pred = rfr.predict(book_features.reshape(1, -1))[0]
            rfr_predictions.append((book, rfr_pred))

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

    return final_cb_recommendations[['work_id', 'title', 'author', 'description', 'image_url', 'similarity_score','predicted_rating','final_score']]

# Hybrid recommendations
@st.cache_data
def get_hy_recommendations(username, work_id):
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
    
    return final_recommendations[['work_id', 'title', 'author', 'description', 'image_url']]


def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()


@st.cache_data
def get_uid(username):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT user_id FROM users WHERE LOWER(username) = LOWER(?)", (username,))
    user_id = cursor.fetchone()
    conn.close()

    return user_id[0] if user_id else None


def insert_user(username, password):
    conn = get_db_connection()
    cursor = conn.cursor()

    username = username.lower()
    password_hash = hash_password(password)
    
    try:
        cursor.execute("SELECT username FROM users WHERE username = ?", (username,))
        existing_user = cursor.fetchone()

        if existing_user:
            return False

        cursor.execute("INSERT INTO users (username, pass_hash) VALUES (?, ?)",
                       (username, password_hash))

        cursor.lexecute("SELECT SCOPE_IDENTITY()")
        user_id = cursor.fetchone()[0]

        conn.commit()

        return user_id
    except pyodbc.IntegrityError:
        return False
    finally:
        conn.close()


def check_credentials(username, password):
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM users WHERE LOWER(username) = LOWER(?) AND pass_hash=?",
                   (username, hash_password(password)))
    user = cursor.fetchone()
    conn.close()

    return user
    
@st.cache_data
def get_goodbooks(limit):
    conn = get_db_connection()
    query = f'SELECT TOP {limit} work_id, title, author, description, genres, image_url FROM books;'
    goodbooks = pd.read_sql(query, conn)
    conn.close()

    # Ensure 'genres' is treated as a string and handle NULL values
    goodbooks['genres'] = goodbooks['genres'].fillna('')
    goodbooks['genres'] = goodbooks['genres'].astype(str)

    return goodbooks

@st.cache_data
def get_existing_rating(username, work_id):
    work_id = int(work_id)
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        SELECT rating FROM ratings WHERE username = ? AND work_id = ?
    ''', (username, work_id))
    exisiting_rating = cursor.fetchone()
    conn.close()
    
    return exisiting_rating[0] if exisiting_rating else None

@st.cache_data
def get_existing_sentiment(username, work_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        SELECT sentiment_score FROM reviews WHERE username = ? AND work_id = ?
    ''', (username, work_id))
    sentiment = cursor.fetchone()
    conn.close()

    return sentiment[0] if sentiment else None

@st.cache_data
def get_user_review(username, work_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        SELECT review_txt FROM reviews WHERE username = ? AND work_id = ?
    ''', (username, work_id))
    existing_review = cursor.fetchone()
    conn.close()

    return existing_review[0] if existing_review else None


def save_rating(rating, username, work_id):
  # Ensure the rating is a float and work_id is an integer
    try:
        rating = float(rating)  # Explicitly cast rating to float
    except ValueError:
        print(f"Error: Rating '{rating}' is not a valid number.")
        return
    
    work_id = int(work_id)  # Explicitly cast work_id to integer
    
    # Print out values and types for debugging purposes
    print(f"Username: {username} (Type: {type(username)})")
    print(f"Rating: {rating} (Type: {type(rating)})")
    print(f"Work ID: {work_id} (Type: {type(work_id)})")
    
    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        # Fetch the user_id from the users table
        cursor.execute("SELECT user_id FROM users WHERE username = ?", (username,))
        user_id_result = cursor.fetchone()

        if user_id_result is None:
            raise ValueError(f"Username '{username}' not found in users table.")
        
        user_id = user_id_result[0]
        print(f"Found user_id: {user_id} for username: {username}")

        # Prepare the SQL statement to insert the rating
        insert_sql = '''
            INSERT INTO ratings (rating, username, work_id, user_id)
            VALUES (?, ?, ?, ?);
        '''
        
        # Debug: Print the SQL query and parameters
        print(f"Executing SQL: {insert_sql}")
        print(f"Parameters: rating={rating}, username={username}, work_id={work_id}, user_id={user_id}")
        
        # Execute the SQL query
        cursor.execute(insert_sql, (rating, username, work_id, user_id))

        # Commit the transaction to save the changes
        conn.commit()
        print(f"Rating {rating} for work_id {work_id} by user {username} saved successfully.")

    except Exception as e:
        # If there's any exception, print the error message
        print(f"Error during query execution: {e}")
    finally:
        # Always close the connection
        conn.close()


def save_review(username, work_id, review_text):
    sentiment_score = analyze_sentiment_vader(review_text)
    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        if review_text.strip():
            review_date = pd.to_datetime("now", utc=True).strftime('%Y-%m-%d')

            # 1. Check if a review already exists for this user and work_id
            cursor.execute("SELECT 1 FROM reviews WHERE username = ? AND work_id = ?", (username, work_id))
            existing_review = cursor.fetchone()

            if existing_review:
                # 2. If a review exists, UPDATE it
                cursor.execute("""
                    UPDATE reviews
                    SET review_txt = ?, review_date = ?, sentiment_score = ?
                    WHERE username = ? AND work_id = ?
                """, (review_text, review_date, sentiment_score, username, work_id))
            else:
                # 3. If no review exists, INSERT a new one
                cursor.execute("""
                    INSERT INTO reviews (username, work_id, review_txt, review_date, sentiment_score)
                    VALUES (?, ?, ?, ?, ?)
                """, (username, work_id, review_text, review_date, sentiment_score))

            conn.commit()
            return True  # Indicate success
        else:
            return False  # Indicate failure (empty review)
    except pyodbc.Error as ex:
        print(f"Database Error: {ex}")
        return False
    finally:
        conn.close()

@st.cache_data   
def get_all_reviews(work_id):
    conn = get_db_connection()
    query = '''
        SELECT r.review_txt, u.username, r.review_date 
        FROM reviews r 
        JOIN users u ON r.username = u.username
        WHERE r.work_id = ? 
        ORDER BY r.review_date DESC
    '''
    reviews_df = pd.read_sql_query(query, conn, params=(work_id,))
    conn.close()
    return reviews_df

@st.cache_data
def get_top_rated_books_from_db():
    conn = get_db_connection()
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
    conn.close()
    return top_rated_books

def get_top_rated_books():
    return get_top_rated_books_from_db()


def analyze_sentiment_vader(review_text):
    sia = SentimentIntensityAnalyzer()
    if not review_text.strip():
        return 0 # Neutral sentiment for empty reviews
    sentiment = sia.polarity_scores(review_text)
    return sentiment['compound']

@st.cache_data
def get_user_rating_count(username):
    conn = get_db_connection()
    query = '''
        SELECT COUNT(DISTINCT work_id) AS rated_books_count
        FROM ratings
        WHERE username = ?
    '''
    params = (username,)
    result = pd.read_sql_query(query, conn, params=params)
    conn.close()

    # Extract the count value from the query result
    rated_books_count = result['rated_books_count'].iloc[0]
    return rated_books_count

@st.cache_data
def get_user_review_count(username):
    conn = get_db_connection()
    query = '''
        SELECT COUNT(DISTINCT work_id) AS review_count
        FROM reviews
        WHERE username = ?
    '''
    result = pd.read_sql(query, conn, params=(username,))
    conn.close()

    return result['review_count'].iloc[0]