import re
import string
import nltk
import requests
import pandas as pd
import numpy as np
import gensim.downloader as gdl
from io import StringIO
from sklearn.preprocessing import StandardScaler
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from gensim.models import KeyedVectors
from bookscout_rs import setup_database, insert_books_from_df, hash_password, insert_users_from_df, insert_ratings_from_df, initialize_nltk

# Extract and load data
books_url = "https://raw.githubusercontent.com/malcolmosh/goodbooks-10k-extended/refs/heads/master/books_enriched.csv"
ratings_url = "https://raw.githubusercontent.com/malcolmosh/goodbooks-10k-extended/refs/heads/master/ratings.csv"

def download_csv(url):
    response = requests.get(url)
    response.raise_for_status()
    return pd.read_csv(StringIO(response.text))

books = download_csv(books_url)
ratings = download_csv(ratings_url)

# Rename book_id column
ratings.rename(columns={'book_id':'work_id'}, inplace=True)

# Remove books from ratings df that do not exist in books df
matched_ids = ratings[ratings['work_id'].isin(books['work_id'])]
ratings['work_id'] = matched_ids['work_id']
ratings.dropna(inplace=True)
ratings['work_id'] = ratings['work_id'].astype('int64')

# Combine books and ratings dfs
goodbooks = books.merge(ratings, how='left', on='work_id')

# Handle missing values in user id and ratings cols
goodbooks['user_id'].fillna(0, inplace=True)
goodbooks['rating'].fillna(0, inplace=True)
goodbooks['user_id'] = goodbooks['user_id'].astype('int64')
goodbooks['rating'] = goodbooks['rating'].astype('int64')

# Drop cols from goodbooks df
goodbooks.drop(columns=['Unnamed: 0', 'index', 'average_rating', 
                        'best_book_id', 'isbn', 'isbn13',
                        'original_publication_year', 'original_title',
                        'pages', 'publishDate', 'authors_2', 'book_id', 
                        'ratings_2', 'books_count', 'ratings_4',
                        'work_text_reviews_count', 'work_ratings_count', 
                        'ratings_count', 'ratings_5', 'ratings_3',
                        'ratings_1', 'language_code'], inplace=True)

# Drop missing values in description col
goodbooks.dropna(subset=['description'], inplace=True)

# Clean titles
goodbooks['title'] = goodbooks['title'].str.replace(r'\s*\(.*?\)\s*', ' ', regex=True).str.strip()

# Remove 'nophoto' urls from image url col
goodbooks['image_url'] = goodbooks['image_url'].str.strip()
pattern = 'https://s.gr-assets.com/assets/nophoto/book/'
goodbooks = goodbooks[~goodbooks['image_url'].str.contains(pattern, case=False, na=False)]

# Clean authors
goodbooks['authors'] = goodbooks['authors'].str.replace(r"[\[\]'']", '', regex=True).str.strip()
goodbooks['authors'] = goodbooks['authors'].apply(lambda x: x.split(',')[0].split(';')[0].strip())

# Normalize descriptions
goodbooks['description'] = goodbooks['description'].str.lower().str.capitalize()

def normalize_description(desc):
    desc = re.sub(r'\.([a-z])', r'. \1', desc)
    sentences = sent_tokenize(desc)
    normalized_sentences = []
    
    for sentence in sentences:
        words = word_tokenize(sentence)
        normalized_words = [word if word.istitle() or word.lower() != 'i' else 'I' for word in words]
        
        if normalized_words:
            normalized_words[0] = normalized_words[0].capitalize()
            normalized_sentences.append(' '.join(normalized_words))
            
    return ' '.join(normalized_sentences)

goodbooks['description'] = goodbooks['description'].apply(normalize_description)

# Prepare description embeddings
translator = str.maketrans('', '', string.punctuation + string.digits)
goodbooks['desc_emb'] = goodbooks['description'].apply(lambda x: x.translate(translator).lower())

nltk.download('stopwords')
en_stopwords = set(stopwords.words('english'))

def filter_description(desc):
    words = desc.split() 
    filtered_words = [w for w in words if w not in en_stopwords and len(w) > 3]
    return ' '.join(filtered_words)

goodbooks['desc_emb'] = goodbooks['desc_emb'].apply(filter_description)

nltk.download('punkt')
goodbooks['desc_emb'] = goodbooks['desc_emb'].apply(word_tokenize)

w2v_path = gdl.load("word2vec-google-news-300", return_path=True)
w2v_model = KeyedVectors.load_word2vec_format(w2v_path, binary=True)

def get_average_embedding(text, model):
    if not text:
        return np.zeros(model.vector_size)
    vectors = [model[t] for t in text if t in model]
    return np.mean(vectors, axis=0) if vectors else np.zeros(model.vector_size)

goodbooks['desc_emb'] = goodbooks['desc_emb'].apply(lambda x: get_average_embedding(x, w2v_model))

# Flatten and standardize description embeddings
flat_embeddings = pd.DataFrame(goodbooks['desc_emb'].tolist())
scaler = StandardScaler()
scaled_embeddings = scaler.fit_transform(flat_embeddings)
goodbooks['scaled_desc_emb'] = list(scaled_embeddings)

# Add usernames col
userid_to_username = {}

def generate_username(user_id):
    return f'user_{user_id}'

for index, row in goodbooks.iterrows():
    user_id = row['user_id']

    if user_id not in userid_to_username:
        username = generate_username(user_id)
        userid_to_username[user_id] = username

goodbooks['username'] = goodbooks['user_id'].map(userid_to_username)

 # Add password col
default_pass = 'default123'
goodbooks['password_hash'] = goodbooks['username'].apply(lambda x: hash_password(default_pass))

# Set up
setup_database()
insert_books_from_df(goodbooks)
uid_mapping = insert_users_from_df(goodbooks)
insert_ratings_from_df(goodbooks, uid_mapping)
initialize_nltk()

print("Database setup and data insertion complete.")