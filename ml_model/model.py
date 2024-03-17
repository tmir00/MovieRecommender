import os
from typing import Union, List, Tuple

import numpy as np
import psycopg2
import pandas as pd

from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import OneHotEncoder, MultiLabelBinarizer

# Declare global variables and macros
COSINE_SIMILARITIES_PATH = '/app/data/cosine_similarities.npy'
MOVIE_IDS_LIST_PATH = '/app/data/movie_ids_list.npy'

db_connection = psycopg2.connect(
    dbname="movielens",
    user="user",
    password="password",
    host="db"
)
cur = db_connection.cursor()

global all_cosine_similarities, movie_ids
if os.path.exists(COSINE_SIMILARITIES_PATH):
    all_cosine_similarities = np.load(COSINE_SIMILARITIES_PATH)
else:
    all_cosine_similarities = None
if os.path.exists(MOVIE_IDS_LIST_PATH):
    movie_ids = np.load(MOVIE_IDS_LIST_PATH)
else:
    all_cosine_similarities = None


###################################### Collaborative Filtering ######################################
def find_highly_rated_movies_for_similar_users(age: int, gender: str, occupation: str, top_k=6) -> Union[None, List[str]]:
    """
    Given age, gender, occupation of a user, find users with similar attributes, find their top-rated movies and
    return the top_k movies across all the movies found.

    :param age: The given user's age.
    :param gender: The given user's gender.
    :param occupation: The given user's occupation.
    :param top_k: The number of movies that are highest rated by similar users to return.
    :return: top k number of movies that are high rated by similar users.
    """
    # Fetch similar users
    similar_users = find_similar_users(age, gender, occupation)
    if not similar_users:
        return None

    movie_ratings = defaultdict(float)
    # Iterate over each list of users starting from the most similar
    for users in similar_users:
        if not users:
            continue
        # Convert user IDs to a comma-separated string
        user_ids_str = ','.join([str(u) for u in users])
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

    # Sort movies by their average rating and return the top k.
    sorted_movies = sorted(movie_ratings.items(), key=lambda x: x[1], reverse=True)
    top_movies = [movie[0] for movie in sorted_movies[:top_k]]
    return top_movies


def find_similar_users(age: int, gender: str, occupation: str) -> Union[None, List[str]]:
    """
    Find users with similar age, gender and occupation to the given parameters.

    :param age: Age of the users to look for.
    :param gender: Gender of the users to look for.
    :param occupation: Occupation of the users to look for.
    :return: The list of similar user_ids or None if there aren't any
    """
    # These queries get similar users in order of most similar to slightly similar.
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

    # Run each of the queries, add most similar users to the start of the list.
    for query in queries:
        temp_users = pd.read_sql_query(query, db_connection)
        results.append(temp_users['user_id'].tolist())

    return results if any(results) else None


###################################### Content-Based Filtering ######################################

def preprocess_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocesses the given DataFrame by filling missing values and splitting specific columns into lists.

    :param df: The DataFrame to preprocess.
    :return: The preprocessed DataFrame.
    """
    # Replace any N/A values in the data with an empty string for the columns we are considering.
    # Preprocess the data into a format that is suitable for our model.
    for column in ['overview', 'production_countries', 'production_companies', 'genres', 'original_language']:
        # If the overview is stored in a list, it extracts it into a string.
        if column == 'overview':
            df[column] = df[column].fillna("").apply(lambda x: ' '.join(x) if isinstance(x, list) else x)
        else:
            # For data types stored as value1|value2, change it to [value1, value2]
            df[column] = df[column].fillna("").apply(lambda x: x.split('|') if column != 'original_language' else x)
    return df


def process_data(df: pd.DataFrame) -> None:
    """
    Processes the data by preprocessing columns, encoding features, and calculating cosine similarities.

    :param df: The DataFrame containing movie data to be processed.
    """
    # Preprocess data into a format that is suitable for our model.
    df_preprocessed = preprocess_columns(df)

    # Transform movie summary to a matrix of TF-IDF features.
    tfidf_vectorizer = TfidfVectorizer(max_features=20)
    overview_tfidf = tfidf_vectorizer.fit_transform(df_preprocessed['overview']).toarray()

    # Initialize MultiLabelBinarizers for encoding the 'genres', 'production_countries', and
    # 'production_companies' columns into binary vectors and encode them.
    mlb_genres = MultiLabelBinarizer()
    mlb_countries = MultiLabelBinarizer()
    mlb_companies = MultiLabelBinarizer()

    genres_encoded = mlb_genres.fit_transform(df_preprocessed['genres'].tolist())
    countries_encoded = mlb_countries.fit_transform(df_preprocessed['production_countries'].tolist())
    companies_encoded = mlb_companies.fit_transform(df_preprocessed['production_companies'].tolist())

    # Initialize OneHotEncoder for encoding the 'original_language' column into binary vectors.
    ohe_language = OneHotEncoder()
    language_encoded = ohe_language.fit_transform(df_preprocessed[['original_language']]).toarray()

    # Combine all the encoded features and TF-IDF vectors into a single matrix representing the complete feature set.
    combined_features = np.hstack([
        genres_encoded,
        countries_encoded,
        companies_encoded,
        language_encoded,
        overview_tfidf
    ])

    # Calculate and save cosine similarities in npy file.
    cosine_similarities = cosine_similarity(combined_features)
    np.save(COSINE_SIMILARITIES_PATH, cosine_similarities)

    # Save movie ids in npy file.
    movie_ids_list = df_preprocessed['id'].tolist()
    np.save(MOVIE_IDS_LIST_PATH, movie_ids_list)


def load_and_compare(movie_id: str) -> List[str]:
    """
    Loads the precomputed cosine similarities and identifies movies similar to the given movie ID.

    :param movie_id: The ID of the movie to find similarities for.
    :return: A list of the most similar movie_id.
    """
    global all_cosine_similarities, movie_ids

    # Load cosine similarities and movie ids
    if all_cosine_similarities is None:
        all_cosine_similarities = np.load(COSINE_SIMILARITIES_PATH)
    if movie_ids is None:
        movie_ids = np.load(MOVIE_IDS_LIST_PATH)

    movie_ids_list = movie_ids.tolist()
    movie_index = movie_ids_list.index(movie_id)
    movie_similarities = all_cosine_similarities[movie_index]

    # Select the top 3 indices from the sorted list, excluding the index of the movie itself
    similar_indices = movie_similarities.argsort()
    similar_movie_ids = [movie_ids_list[i] for i in similar_indices[-4:-1]]

    return similar_movie_ids[-1:]


def train():
    """
    Trains the model by processing data from the database and saving the computed features and similarities.
    """
    query = f"""
        SELECT * FROM movies
    """
    global all_cosine_similarities, movie_ids
    # Execute the SQL query and load all movie data into a pandas DataFrame.
    df = pd.read_sql_query(query, db_connection)

    # Process this data.
    process_data(df)

    # Load trained data into global variables for easier access for subsequent API calls.
    all_cosine_similarities = np.load(COSINE_SIMILARITIES_PATH)
    movie_ids = np.load(MOVIE_IDS_LIST_PATH)


def run_model(user_id) -> Tuple[List[int], bool]:
    """
    Runs the recommendation model for a given user ID to find recommended movies.

    :param user_id: The ID of the user for whom to run the model.
    :return: A tuple containing a list of recommended movie IDs and a boolean indicating if the user has an account.
    """
    # If the model has not been trained already, train it.
    if not os.path.exists(COSINE_SIMILARITIES_PATH) or not os.path.exists(MOVIE_IDS_LIST_PATH):
        train()

    # Find the highest rated movies amongst most similar users.
    account = True
    user_query = f"""
        SELECT age, gender, occupation FROM users
        WHERE user_id = %s
    """
    cur.execute(user_query, (user_id,))
    user_results = cur.fetchall()
    best_movies = []

    # If the given user has no account, store this.
    if not user_results:
        account = False
    else:
        best_movies = find_highly_rated_movies_for_similar_users(user_results[0][0], user_results[0][1], user_results[0][2])

    # Fetch the highest rated movies of the given user, then find similar movies to it.
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

    # Select the most popular movies with a large vote count.
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

    # Use all the previous obtained movies and return them in one list.
    return best_movies, account
