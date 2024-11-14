#!/usr/bin/env python
# coding: utf-8

import os
import pickle
import sqlite3
import hashlib
import pandas as pd
import numpy as np
import hnswlib
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split


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
        user_id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT NOT NULL,
        pass_hash TEXT NOT NULL
    );
    ''')
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS ratings (
        rating_id INTEGER PRIMARY KEY AUTOINCREMENT,
        rating INTEGER,
        user_id INTEGER,
        work_id INTEGER,
        UNIQUE(user_id, work_id),
        FOREIGN KEY (user_id) REFERENCES users(user_id),
        FOREIGN KEY (work_id) REFERENCES books(work_id)
    );
    ''')
        
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS reviews (
        review_id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        work_id INTEGER,
        review_txt TEXT,
        review_date DATETIME,
        FOREIGN KEY (user_id) REFERENCES users(user_id),
        FOREIGN KEY (work_id) REFERENCES books(work_id)
    );
    ''')
    
    conn.commit()
    conn.close()


def insert_books_from_df(df):
    conn = sqlite3.connect('bookscout.db')
    cursor = conn.cursor()
    for _, row in df.iterrows():
        # Convert embeddings to bytes for SQLite compatability.
        embeddings = np.array(row['scaled_desc_emb']).tobytes() if 'scaled_desc_emb' in row else None
        cursor.execute('''
        INSERT OR IGNORE INTO books (work_id, title, author, description, genres, desc_emb, image_url) VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (row['work_id'], row['title'], row['authors'], row['description'], row['genres'], embeddings, row['image_url']))
    conn.commit()
    conn.close()


def insert_users_from_df(df):
    conn = sqlite3.connect('bookscout.db')
    cursor = conn.cursor()
    
    uid_mapping = {}

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

        uid_mapping[username.lower()] = user_id

        conn.commit()
        conn.close()
        return uid_mapping


def insert_ratings_from_df(df, uid_mapping):
    conn = sqlite3.connect('bookscout.db')
    cursor = conn.cursor()

    for _, row in df.iterrows():
        username = row['username']
        user_id = uid_mapping.get(username.lower())

        if user_id is not None:
            work_id = row['work_id']
            rating = row['rating']
            
            cursor.execute('''
            INSERT OR REPLACE INTO ratings (user_id, work_id, rating) VALUES (?, ?, ?)
            ''', (user_id, work_id, rating))

    conn.commit()
    conn.close()


def reset_database():
    if os.path.exists('bookscout.db'):
        os.remove('bookscout.db')


def get_cf_data():
    conn = sqlite3.connect('bookscout.db')
    ratings_df = pd.read_sql_query('SELECT * FROM ratings', conn)
    conn.close()

    return ratings_df


def prepare_cf_data(ratings_df):
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(ratings_df[['user_id', 'work_id', 'rating']], reader)

    return data


def load_svd():
    try:
        with open('svd.pkl', 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None


def train_svd(data):
    svd = load_svd()
    if svd:
        return svd
    
    trainset, testset = train_test_split(data, test_size=0.25)
    svd = SVD(n_factors=150, n_epochs=40, lr_all=0.011, reg_all=0.006)
    svd.fit(trainset)
    
    with open('svd.pkl', 'wb') as f:
        pickle.dump(svd, f)

    return svd


def get_cf_recommendations(user_id):
    ratings_df = get_cf_data()
    data = prepare_cf_data(ratings_df)
    model = train_svd(data)
    
    all_books = ratings_df['work_id'].unique()
    rated_books = ratings_df[(ratings_df['user_id'] == user_id) & (ratings_df['rating'] > 0)]['work_id']
    predictions = []
    
    for book in all_books:
        if book not in rated_books:
            pred = model.predict(user_id, book)
            predictions.append((book, pred.est))
    
    predictions.sort(key=lambda x: x[1], reverse=True)
    top_cf_recommendations = predictions[:5]

    return top_cf_recommendations

def get_cb_data():
    conn = sqlite3.connect('bookscout.db')
    books_df = pd.read_sql_query('SELECT work_id, title, author, description, desc_emb, image_url FROM books', conn)
    conn.close()

    return books_df


def load_embeddings(books_df):
    embeddings_float = []

    for embedding_bytes in books_df['desc_emb']:
        if embedding_bytes is not None:
            try:
                emb = np.frombuffer(embedding_bytes, dtype=np.float32)
                embeddings_float.append(emb)
            except Exception as e:
                print(f"Error converting bytes to array: {e}")
                embeddings_float.append(None)
        else:
            embeddings_float.append(None)

    embeddings_float = [emb for emb in embeddings_float if emb is not None]
    
    if not embeddings_float:
        raise ValueError("No valid embeddings found.")

    return np.array(embeddings_float, dtype=np.float32)


def get_cb_recommendations(work_id, k=10):
    books_df = get_cb_data()
    embeddings_float = load_embeddings(books_df)
    
    if embeddings_float.size == 0:
        raise ValueError("No valid embeddings found.")
    
    dim = embeddings_float.shape[1]
    num_entries = embeddings_float.shape[0]
    hnsw_index = hnswlib.Index(space='cosine', dim=dim)
    hnsw_index.init_index(max_elements=num_entries, ef_construction=300, M=32)
    hnsw_index.add_items(embeddings_float, ids=np.arange(num_entries))
    hnsw_index.set_ef(300)

    book_index = books_df[books_df['work_id'] == work_id].index

    if book_index.size == 0:
        raise ValueError("No valid book index found.")

    query_embedding = embeddings_float[book_index[0]].reshape(1, -1)
    labels, distances = hnsw_index.knn_query(query_embedding, k=k)

    valid_labels = [label for label in labels[0] if label < len(books_df)]
    if not valid_labels:
        print("No valid recommendations found.")
        return []

    top_cb_recommendations = books_df.iloc[valid_labels][['work_id', 'title']]
    top_cb_recommendations = top_cb_recommendations[top_cb_recommendations['work_id'] != work_id]
    top_cb_recommendations = top_cb_recommendations.drop_duplicates(subset='work_id')[['work_id', 'title']][:5]

    return top_cb_recommendations


def get_hy_recommendations(user_id, work_id, cf_weight=0.5, cb_weight=0.5):
    cf_recommendations = get_cf_recommendations(user_id)
    cf_df = pd.DataFrame(cf_recommendations, columns=['work_id', 'predicted_rating'])
    
    cb_recommendations = get_cb_recommendations(work_id)
    cb_df = cb_recommendations.copy()
    cb_df['predicted_rating'] = 1.0
    
    hy_recommendations = pd.concat([cf_df.rename(columns={'predicted_rating': 'cf_rating'}),
                                 cb_df.rename(columns={'predicted_rating': 'cb_rating'})],
                                 ignore_index=True)
    hy_recommendations = hy_recommendations.groupby('work_id').agg({'cf_rating': 'sum', 'cb_rating': 'sum'}).reset_index()
    hy_recommendations['final_rating'] = (hy_recommendations['cf_rating'] * cf_weight) + (hy_recommendations['cb_rating'] * cb_weight)
    
    top_hy_recommendations = hy_recommendations.sort_values(by='final_rating', ascending=False).head(5)
    books_df = get_cb_data()
    final_recommendations = top_hy_recommendations.merge(books_df[['work_id', 'title', 'author', 'description', 'image_url']], on='work_id', how='left')
    
    return final_recommendations[['work_id', 'title', 'author', 'description', 'image_url']]


def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()


def get_uid(username):
    conn = sqlite3.connect('bookscout.db')
    cursor = conn.cursor()
    cursor.execute("SELECT user_id FROM users WHERE LOWER(username) = LOWER(?)", (username,))
    user_id = cursor.fetchone()
    conn.close()

    return user_id[0] if user_id else None


def insert_user(username, password):
    conn = sqlite3.connect('bookscout.db')
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
        conn.commit()

        return cursor.lastrowid
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()


def check_credentials(username, password):
    conn = sqlite3.connect('bookscout.db')
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM users WHERE LOWER(username) = LOWER(?) AND pass_hash=?",
                   (username, hash_password(password)))
    user = cursor.fetchone()
    conn.close()

    return user
    

def get_goodbooks():
    conn = sqlite3.connect('bookscout.db')
    query = "SELECT work_id, title, author, description, image_url FROM books;"
    goodbooks = pd.read_sql(query, conn)
    conn.close()

    return goodbooks


def get_existing_rating(user_id, work_id):
    conn = sqlite3.connect('bookscout.db')
    cursor = conn.cursor()
    cursor.execute('''
        SELECT rating FROM ratings WHERE user_id = ? AND work_id = ?
    ''', (user_id, work_id))
    exisiting_rating = cursor.fetchone()
    conn.close()
    
    return exisiting_rating[0] if exisiting_rating else None


def get_user_review(user_id, work_id):
    conn = sqlite3.connect('bookscout.db')
    cursor = conn.cursor()
    cursor.execute('''
        SELECT review_txt FROM reviews WHERE user_id = ? AND work_id = ?
    ''', (user_id, work_id))
    existing_review = cursor.fetchone()
    conn.close()

    return existing_review[0] if existing_review else None


def save_rating(user_id, work_id, rating):
    conn = sqlite3.connect('bookscout.db')
    cursor = conn.cursor()
    cursor.execute('''
        INSERT OR REPLACE INTO ratings (user_id, work_id, rating) VALUES (?, ?, ?)
    ''', (user_id, work_id, rating))
    conn.commit()
    conn.close()


def save_review(user_id, work_id, review_text):
    conn = sqlite3.connect('bookscout.db')
    cursor = conn.cursor()
    if review_text.strip(): 
        review_date = pd.to_datetime("now")
        review_date_str = review_date.strftime('%Y-%m-%d')
        cursor.execute('''
            INSERT OR REPLACE INTO reviews (user_id, work_id, review_txt, review_date) VALUES (?, ?, ?, ?)
        ''', (user_id, work_id, review_text, review_date_str))
    conn.commit()
    conn.close()

def get_all_reviews(work_id):
    conn = sqlite3.connect('bookscout.db')
    query = '''
        SELECT r.review_txt, u.username, r.review_date 
        FROM reviews r 
        JOIN users u ON r.user_id = u.user_id
        WHERE r.work_id = ? 
        ORDER BY r.review_date DESC
    '''
    reviews_df = pd.read_sql_query(query, conn, params=(work_id,))
    conn.close()
    return reviews_df

def get_top_rated_books():
    conn = sqlite3.connect('bookscout.db')
    cursor = conn.cursor()
    cursor.execute('''
        SELECT b.work_id, b.title, b.author, b.image_url, AVG(r.rating) as avg_rating
        FROM books b
        JOIN ratings r ON b.work_id = r.work_id
        GROUP BY b.work_id
        ORDER BY avg_rating DESC
        LIMIT 10
    ''')
    top_rated_books = pd.DataFrame(cursor.fetchall(), columns=['work_id', 'title', 'author', 'image_url', 'avg_rating'])
    conn.close()
    return top_rated_books