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
        genres TEXT,
        embeddings BLOB,
        image_url TEXT
    )
    ''')
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        user_id INTEGER PRIMARY KEY,
        username TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL
    )
    ''')
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS ratings (
        rating INTEGER PRIMARY KEY,
        user_id INTEGER,
        work_id INTEGER,
        FOREIGN KEY (work_id) REFERENCES books(work_id)
    )
    ''')
    
    conn.commit()
    conn.close()

def insert_books_from_df(df):
    conn = sqlite3.connect('bookscout.db')
    cursor = conn.cursor()
    for _, row in df.iterrows():
        # Convert embeddings to bytes for SQLite compatability.
        embeddings = np.array(row['scaled_embeddings']).tobytes() if 'scaled_embeddings' in row else None
        cursor.execute('''
        INSERT OR IGNORE INTO books (work_id, title, author, genres, embeddings) VALUES (?, ?, ?, ?, ?)
        ''', (row['work_id'], row['title'], row['authors'], row['genres'], embeddings))
    conn.commit()
    conn.close()

def insert_ratings_from_df(df):
    conn = sqlite3.connect('bookscout.db')
    cursor = conn.cursor()
    for _, row in df.iterrows():
        cursor.execute('''
        INSERT OR IGNORE INTO ratings (user_id, work_id, rating) VALUES (?, ?, ?)
        ''', (row['user_id'], row['work_id'], row['rating']))
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
    books_df = pd.read_sql_query('SELECT work_id, title, embeddings FROM books', conn)
    conn.close()
    
    return books_df

def load_embeddings(books_df):
    # Convert the embeddings from bytes back to numpy arrays
    embeddings = []
    
    for embedding_bytes in books_df['embeddings']:
        if embedding_bytes is not None:
            emb = np.frombuffer(embedding_bytes, dtype=np.float32)
            embeddings.append(emb)
        else:
            embeddings.append(None)
    
    embeddings = [emb for emb in embeddings if emb is not None]
    
    if not embeddings:
        raise ValueError("No valid embeddings found.")
    
    first_embedding_shape = embeddings[0].shape
    for emb in embeddings:
        if emb.shape != first_embedding_shape:
            raise ValueError("Inconsistent embedding shapes found.")   
        
    embeddings_float = np.array(embeddings, dtype=np.float32)
    embeddings_object = np.array([np.frombuffer(embedding_bytes, dtype=np.float32) 
                                  if embedding_bytes else None for embedding_bytes 
                                  in books_df['embeddings']], dtype='object')    
            
    return embeddings_float, embeddings_object

def get_cb_recommendations(work_id, k=10):
    books_df = get_cb_data()
    embeddings_float, embeddings_object = load_embeddings(books_df)
    
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

    # Use book_index[0] to get the scalar index
    query_embedding = embeddings_object[book_index[0]].reshape(1, -1)

    labels, distances = hnsw_index.knn_query(query_embedding, k=k)

    # Check for valid labels
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
    books_df = fetch_cb_data()
    final_recommendations = top_hy_recommendations.merge(books_df[['work_id', 'title']], on='work_id', how='left')
    
    return final_recommendations[['work_id', 'title']]



def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def get_uid(username):
    conn = sqlite3.connect('bookscout.db')
    cursor = conn.cursor()
    cursor.execute("SELECT user_id FROM users WHERE LOWER(username) = LOWER(?)", (username,))
    user_id = cursor.fetchone()
    conn.close()

    return user_id[0] if user_id else None

def get_next_uid(cursor):
    cursor.execute("SELECT MAX(user_id) FROM ratings")
    max_id = cursor.fetchone()[0]

    return max_id + 1 if max_id is not None else 1

def insert_user(username, password):
    conn = sqlite3.connect('bookscout.db')
    cursor = conn.cursor()

    password_hash = hash_password(password)
    
    try:
        cursor.execute("SELECT username FROM users WHERE LOWER(username) = LOWER(?)", (username,))
        existing_user = cursor.fetchone()

        if existing_user:
            return False
        
        user_id = get_next_uid(cursor)

        cursor.execute("INSERT INTO users (user_id, username, password_hash) VALUES (?, ?, ?)",
                       (user_id, username.lower(), password_hash))
        conn.commit()

        return user_id
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def check_credentials(username, password):
    conn = sqlite3.connect('bookscout.db')
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM users WHERE LOWER(username) = LOWER(?) AND password_hash=?",
                   (username, hash_password(password)))
    user = cursor.fetchone()
    conn.close()

    return user
    


