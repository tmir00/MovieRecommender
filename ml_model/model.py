import os

import numpy as np
import psycopg2
import pandas as pd

from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import OneHotEncoder, MultiLabelBinarizer

db_connection = psycopg2.connect(
    dbname="movielens",
    user="user",
    password="password",
    host="localhost"  # or "db" if running inside a Docker network
)

cur = db_connection.cursor()

global all_cosine_similarities, movie_ids

if os.path.exists('./cosine_similarities.npy'):
    all_cosine_similarities = np.load('./cosine_similarities.npy')

if os.path.exists('./movie_ids_list.npy'):
    movie_ids = np.load('./movie_ids_list.npy')


###################################### Collaborative Filtering ######################################
def find_highly_rated_movies_for_similar_users(age, gender, occupation):
    # Fetch similar users
    similar_users = find_similar_users(age, gender, occupation)
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
        temp_movies = pd.read_sql_query(query, db_connection)
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
def find_similar_users(age, gender, occupation):
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
        temp_users = pd.read_sql_query(query, db_connection)
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


def load_and_compare(movie_id):
    global all_cosine_similarities, movie_ids
    all_cosine_similarities = np.load('./cosine_similarities.npy')
    movie_ids = np.load('./movie_ids_list.npy')

    movie_ids_list = movie_ids.tolist()
    movie_index = movie_ids_list.index(movie_id)
    movie_similarities = all_cosine_similarities[movie_index]

    similar_indices = movie_similarities.argsort()
    similar_movie_ids = [movie_ids_list[i] for i in similar_indices[-4:-1] if i != movie_index]

    return similar_movie_ids[-1:]


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
    account = True

    user_query = f"""
        SELECT age, gender, occupation FROM users
        WHERE user_id = %s
    """
    cur.execute(user_query, (user_id,))
    user_results = cur.fetchall()
    best_movies = []

    if not user_results:
        account = False
    else:
        best_movies = find_highly_rated_movies_for_similar_users(user_results[0][0], user_results[0][1], user_results[0][2])

    query = """
            SELECT movie, rating FROM ratings
            WHERE user_id = %s AND rating >= 4 
            ORDER BY rating DESC
            LIMIT 3
        """

    cur.execute(query, (user_id,))
    results = cur.fetchall()

    for movie in range(len(results)):
        best_movies += load_and_compare(results[movie][0])

    query_popular = """
            SELECT id, vote_average, vote_count FROM movies
            WHERE vote_count > 200 
            ORDER BY vote_average DESC 
            LIMIT 10
        """
    cur.execute(query_popular)
    results = cur.fetchall()
    sorted_results = sorted(results, key=lambda x: x[1])

    for i in range(len(sorted_results)):
        if len(best_movies) == 10:
            break
        if sorted_results[i][0] not in best_movies:
            best_movies.append(sorted_results[i][0])

    return best_movies, account

for i in range(3):
    result = run_model(i)
    print("Account: {}, Movies: {}".format(result[1], result[0]))
