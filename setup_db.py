import string
import nltk
import pandas as pd
import numpy as np
import gensim.downloader as gdl
from sklearn.preprocessing import StandardScaler
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim.models import KeyedVectors
from bookscout_rs import setup_database, insert_books_from_df, insert_ratings_from_df


books = pd.read_csv('data/books_enriched.csv')
ratings = pd.read_csv('data/ratings.csv')

ratings.rename(columns={'book_id':'work_id'}, inplace=True)

matched_ids = ratings[ratings['work_id'].isin(books['work_id'])]
ratings['work_id'] = matched_ids['work_id']
ratings.dropna(inplace=True)
ratings['work_id'] = ratings['work_id'].astype('int64')

goodbooks = books.merge(ratings, how='left', on='work_id')

goodbooks['user_id'].fillna(0, inplace=True)
goodbooks['rating'].fillna(0, inplace=True)
goodbooks['user_id'] = goodbooks['user_id'].astype('int64')
goodbooks['rating'] = goodbooks['rating'].astype('int64')

goodbooks.drop(columns=['Unnamed: 0', 'index', 'average_rating', 
                        'best_book_id', 'isbn', 'isbn13',
                        'original_publication_year', 'original_title',
                        'pages', 'publishDate', 'authors_2', 'book_id', 
                        'ratings_2', 'books_count', 'ratings_4',
                        'work_text_reviews_count', 'work_ratings_count', 
                        'ratings_count', 'ratings_5', 'ratings_3',
                        'ratings_1', 'language_code'], inplace=True)

goodbooks.dropna(subset=['description'], inplace=True)

goodbooks['title'] = goodbooks['title'].str.replace(r'\s*\(.*?\)\s*', ' ', regex=True).str.strip()
goodbooks['authors'] = goodbooks['authors'].str.replace(r"[\[\]'']", '', regex=True).str.strip()
goodbooks['authors'] = goodbooks['authors'].apply(lambda x: x.split(',')[0].split(';')[0].strip())

translator = str.maketrans('', '', string.punctuation + string.digits)
goodbooks['description'] = goodbooks['description'].apply(lambda x: x.translate(translator).lower())

nltk.download('stopwords')
en_stopwords = set(stopwords.words('english'))

def filter_description(desc):
    words = desc.split() 
    filtered_words = [w for w in words if w not in en_stopwords and len(w) > 3]
    return ' '.join(filtered_words)

goodbooks['description'] = goodbooks['description'].apply(filter_description)

nltk.download('punkt')
goodbooks['description'] = goodbooks['description'].apply(word_tokenize)

w2v_path = gdl.load("word2vec-google-news-300", return_path=True)
w2v_model = KeyedVectors.load_word2vec_format(w2v_path, binary=True)

def get_average_embedding(text, model):
    if not text:
        return np.zeros(model.vector_size)
    vectors = [model[t] for t in text if t in model]
    return np.mean(vectors, axis=0) if vectors else np.zeros(model.vector_size)

goodbooks['embeddings'] = goodbooks['description'].apply(lambda x: get_average_embedding(x, w2v_model))

flat_embeddings = pd.DataFrame(goodbooks['embeddings'].tolist())

scaler = StandardScaler()
scaled_embeddings = scaler.fit_transform(flat_embeddings)

goodbooks['scaled_embeddings'] = list(scaled_embeddings)

goodbooks['image_url'] = goodbooks['image_url'].str.strip()

pattern = 'https://s.gr-assets.com/assets/nophoto/book/'
goodbooks = goodbooks[~goodbooks['image_url'].str.contains(pattern, case=False, na=False)]

setup_database()
insert_books_from_df(goodbooks)
insert_ratings_from_df(goodbooks)

print("Database setup and data insertion complete.")