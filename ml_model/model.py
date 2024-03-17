import os

import numpy as np
import psycopg2
import pandas as pd
from collections import defaultdict

from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MultiLabelBinarizer
import faiss
from scipy.sparse import hstack, save_npz, load_npz

db_connection = psycopg2.connect(
    dbname="movielens",
    user="user",
    password="password",
    host="localhost"  # or "db" if running inside a Docker network
)


global all_cosine_similarities, movie_ids

if os.path.exists('./cosine_similarities.npy'):
    all_cosine_similarities = np.load('./cosine_similarities.npy')

if os.path.exists('./movie_ids_list.npy'):
    movie_ids = np.load('./movie_ids_list.npy')

###################################### Collaborative Filtering ######################################
def find_highly_rated_movies_for_similar_users(age, gender, occupation, connection):
    # Fetch similar users
    similar_users = find_similar_users(age, gender, occupation, connection)
    if not similar_users:
        return None

    movie_ratings = defaultdict(float)
    # Iterate over each list of users starting from the most similar
    for users in similar_users:
        if not users:
            continue
        user_ids_str = ','.join([str(u) for u in users])  # Convert user IDs to a comma-separated string
        query = f"""
            SELECT movie, AVG(rating) as avg_rating
            FROM ratings
            WHERE user_id IN ({user_ids_str})
            GROUP BY movie
            ORDER BY avg_rating DESC
            LIMIT 3
        """
        temp_movies = pd.read_sql_query(query, connection)
        for index, row in temp_movies.iterrows():
            movie_id = row['movie']
            avg_rating = row['avg_rating']
            movie_ratings[movie_id] = avg_rating

        # If we already have 3 movies, no need to continue
        if len(movie_ratings) >= 6:
            break

    # Sort movies by their average rating and return the top 6
    sorted_movies = sorted(movie_ratings.items(), key=lambda x: x[1], reverse=True)
    top_movies = [movie[0] for movie in sorted_movies[:6]]
    return top_movies

# Function to find similar users based on the previous criteria
def find_similar_users(age, gender, occupation, connection):
    queries = [
        f"""
        SELECT user_id
        FROM users
        WHERE ABS(age - {age}) <= 5
        AND gender = '{gender}'
        AND occupation = '{occupation}'
        """,
        f"""
        SELECT user_id
        FROM users
        WHERE ABS(age - {age}) <= 5
        AND gender = '{gender}'
        """,
        f"""
        SELECT user_id
        FROM users
        WHERE gender = '{gender}'
        """
    ]

    results = []

    for query in queries:
        temp_users = pd.read_sql_query(query, connection)
        results.append(temp_users['user_id'].tolist())

    return results if any(results) else None


###################################### Content-Based Filtering ######################################

def preprocess_columns(df):
    """
    Fill NA values and split genres, production countries, and production companies into lists.
    """
    for column in ['overview', 'production_countries', 'production_companies', 'genres', 'original_language']:
        if column == 'overview':
            df[column] = df[column].fillna("").apply(lambda x: ' '.join(x) if isinstance(x, list) else x)
        else:
            df[column] = df[column].fillna("").apply(lambda x: x.split('|') if column != 'original_language' else x)

    return df


def process_data(df):
    df_preprocessed = preprocess_columns(df)

    tfidf_vectorizer = TfidfVectorizer(max_features=20)
    overview_tfidf = tfidf_vectorizer.fit_transform(df_preprocessed['overview']).toarray()

    mlb_genres = MultiLabelBinarizer()
    mlb_countries = MultiLabelBinarizer()
    mlb_companies = MultiLabelBinarizer()

    genres_encoded = mlb_genres.fit_transform(df_preprocessed['genres'].tolist())
    countries_encoded = mlb_countries.fit_transform(df_preprocessed['production_countries'].tolist())
    companies_encoded = mlb_companies.fit_transform(df_preprocessed['production_companies'].tolist())

    ohe_language = OneHotEncoder()
    language_encoded = ohe_language.fit_transform(df_preprocessed[['original_language']]).toarray()

    combined_features = np.hstack([
        genres_encoded,
        countries_encoded,
        companies_encoded,
        language_encoded,
        overview_tfidf
    ])

    cosine_similarities = cosine_similarity(combined_features)
    np.save('./cosine_similarities.npy', cosine_similarities)

    movie_ids_list = df_preprocessed['id'].tolist()
    np.save('./movie_ids_list.npy', movie_ids_list)


def load_and_compare(cosine_similarities, movies, movie_id):

    movie_ids_list = movies.tolist()
    movie_index = movie_ids_list.index(movie_id)
    movie_similarities = cosine_similarities[movie_index]

    print(len(movie_similarities))
    similar_indices = movie_similarities.argsort()
    similar_movie_ids = [movie_ids_list[i] for i in similar_indices[-4:-1] if i != movie_index]

    return similar_movie_ids[-2:]

def train():
    query = f"""
        SELECT * FROM movies
    """
    global all_cosine_similarities, movie_ids
    df = pd.read_sql_query(query, db_connection)
    process_data(df)
    all_cosine_similarities = np.load('./cosine_similarities.npy')
    movie_ids = np.load('./movie_ids_list.npy')

def run_model(user_id):
    # train()
    results = load_and_compare(all_cosine_similarities, movie_ids, 'captain+america+the+first+avenger+2011')


