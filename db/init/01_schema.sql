-- Drop the tables if they already exist
DROP TABLE IF EXISTS movies;
DROP TABLE IF EXISTS users;
DROP TABLE IF EXISTS ratings;

-- Create the movies table
CREATE TABLE movies (
    id TEXT PRIMARY KEY,
    tmdb_id TEXT,
    imdb_id TEXT,
    title TEXT,
    original_title TEXT,
    adult BOOLEAN,
    budget INT,
    genres TEXT,
    homepage TEXT,
    original_language TEXT,
    overview TEXT,
    popularity NUMERIC,
    poster_path TEXT,
    production_companies TEXT,
    production_countries TEXT,
    release_date DATE,
    revenue BIGINT,
    runtime NUMERIC,
    spoken_languages TEXT,
    status TEXT,
    vote_average NUMERIC,
    vote_count NUMERIC
);

-- Create the users table
CREATE TABLE users (
    user_id INT PRIMARY KEY,
    age INT,
    occupation TEXT,
    gender CHAR(1)
);

-- Create the ratings table
CREATE TABLE ratings (
    timestamp TEXT,
    user_id INT REFERENCES users(user_id),
    movie TEXT,
    rating INT
);

-- Create a temporary table for ratings
CREATE TABLE temp_ratings (
    timestamp TEXT,
    user_id INT,
    movie TEXT,
    rating INT
);

-- Create a temporary table for movies
CREATE TABLE temp_movies (
    id TEXT,
    tmdb_id TEXT,
    imdb_id TEXT,
    title TEXT,
    original_title TEXT,
    adult BOOLEAN,
    budget INT,
    genres TEXT,
    homepage TEXT,
    original_language TEXT,
    overview TEXT,
    popularity NUMERIC,
    poster_path TEXT,
    production_companies TEXT,
    production_countries TEXT,
    release_date DATE,
    revenue BIGINT,
    runtime NUMERIC,
    spoken_languages TEXT,
    status TEXT,
    vote_average NUMERIC,
    vote_count NUMERIC
);


