-- Create the movies table
CREATE TABLE movies (
    movie_id INT PRIMARY KEY,
    title TEXT NOT NULL,
    genres TEXT NOT NULL
);

-- Create ratings table
CREATE TABLE ratings (
    rating_id SERIAL PRIMARY KEY,
    user_id INT,
    movie_id INT REFERENCES movies(movie_id) ON DELETE CASCADE,
    rating FLOAT,
    timestamp BIGINT
);

-- Create tags table
CREATE TABLE tags (
    tag_id SERIAL PRIMARY KEY,
    user_id INT,
    movie_id INT REFERENCES movies(movie_id) ON DELETE CASCADE,
    tag TEXT,
    timestamp BIGINT
);

-- Create links table
CREATE TABLE links (
    movie_id INT PRIMARY KEY REFERENCES movies(movie_id) ON DELETE CASCADE,
    imdb_id TEXT,
    tmdb_id TEXT
);

