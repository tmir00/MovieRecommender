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

global mlb_countries, mlb_companies, ohe_language, tfidf

def sample_and_preprocess_data(df, n_samples):
    global mlb_countries, mlb_companies, ohe_language, tfidf
    df_sampled = df.sample(n=n_samples)

    relevant_attributes = df_sampled[
        ['genres', 'production_countries', 'production_companies', 'original_language', 'overview']]

    mlb_countries = MultiLabelBinarizer()
    mlb_companies = MultiLabelBinarizer()
    ohe_language = OneHotEncoder()

    relevant_attributes['overview'] = relevant_attributes['overview'].fillna("")
    relevant_attributes['production_countries'] = relevant_attributes['production_countries'].fillna("")
    relevant_attributes['production_companies'] = relevant_attributes['production_companies'].fillna("")
    relevant_attributes['genres'] = relevant_attributes['genres'].fillna("")
    relevant_attributes['original_language'] = relevant_attributes['original_language'].fillna("")

    relevant_attributes['production_countries'] = relevant_attributes['production_countries'].apply(
        lambda x: x.split('|'))
    relevant_attributes['production_companies'] = relevant_attributes['production_companies'].apply(
        lambda x: x.split('|'))
    relevant_attributes['genres'] = relevant_attributes['genres'].apply(lambda x: x.split('|'))

    countries_encoded = mlb_countries.fit_transform(relevant_attributes['production_countries'])
    companies_encoded = mlb_companies.fit_transform(relevant_attributes['production_companies'])
    genres_encoded = mlb_companies.fit_transform(relevant_attributes['genres'])
    language_encoded = ohe_language.fit_transform(relevant_attributes[['original_language']])

    tfidf = TfidfVectorizer()
    overview_tfidf = tfidf.fit_transform(relevant_attributes['overview']).toarray()

    features = np.hstack((countries_encoded, companies_encoded, genres_encoded, language_encoded.toarray(), overview_tfidf))

    np.save('preprocessed_data.npy', features)


def transform_new_data(df_new):
    """
    Transform new data using the same preprocessing as before but without fitting the transformers.
    This function is a placeholder; actual implementation should use transform method of fitted preprocessors.
    """
    df_new['overview'] = df_new['overview'].fillna("")
    df_new['production_countries'] = df_new['production_countries'].fillna("")
    df_new['production_companies'] = df_new['production_companies'].fillna("")
    df_new['genres'] = df_new['genres'].fillna("")
    df_new['original_language'] = df_new['original_language'].fillna("")

    countries_encoded = mlb_countries.transform(df_new['production_countries'].apply(lambda x: x.split('|')))
    companies_encoded = mlb_companies.transform(df_new['production_companies'].apply(lambda x: x.split('|')))
    genres_encoded = mlb_companies.transform(df_new['genres'].apply(lambda x: x.split('|')))
    language_encoded = ohe_language.transform(df_new[['original_language']])

    overview_tfidf = tfidf.transform(df_new['overview']).toarray()

    features_new = np.hstack((countries_encoded, companies_encoded, genres_encoded, language_encoded.toarray(), overview_tfidf))
    return features_new


def load_and_compare(movie_ids, df_movies):
    features = np.load('preprocessed_data.npy')
    df_movies_selected = df_movies[df_movies['id'].isin(movie_ids)]
    features_selected = transform_new_data(df_movies_selected)
    similarities = cosine_similarity(features_selected, features)

    similar_movies_ids = []
    for index, movie_similarity in enumerate(similarities):
        top_similar_indices = movie_similarity.argsort()[-3:-1]  # excluding the last one because it's the movie itself
        similar_movies_ids.append(df_movies.iloc[top_similar_indices]['id'].values)

    return similar_movies_ids



sql_query = f"""
SELECT * FROM movies
ORDER BY RANDOM()
"""
df_movies = pd.read_sql_query(sql_query, db_connection)
sample_and_preprocess_data(df_movies, 1000)
movie_ids = ['the+year+my+voice+broke+1987', 'north+country+2005', 'an+honest+liar+2014']
similar_movies_ids = load_and_compare(movie_ids, df_movies)
print(similar_movies_ids)

