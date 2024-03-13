-- Import data into the movies table
-- Copy data into the temporary table
COPY temp_movies(
  id, tmdb_id, imdb_id, title, original_title, adult,
  budget, genres, homepage, original_language, overview, popularity, poster_path,
  production_companies, production_countries, release_date,
  revenue, runtime, spoken_languages, status,
  vote_average, vote_count
)
FROM '/docker-entrypoint-initdb.d/db_data/movies.csv' DELIMITER ',' CSV HEADER;

-- Insert data from the temporary table into the actual table, ignoring duplicate ids
INSERT INTO movies
SELECT DISTINCT ON (id) *
FROM temp_movies
WHERE id IS NOT NULL
ON CONFLICT (id) DO NOTHING;

-- Drop the temporary table
DROP TABLE temp_movies;

-- Import data into the users table
COPY users(user_id, age, occupation, gender)
FROM '/docker-entrypoint-initdb.d/db_data/users.csv' DELIMITER ',' CSV HEADER;

-- Import data into the ratings table
-- Copy data into the temporary table
COPY temp_ratings(timestamp, user_id, movie, rating)
FROM '/docker-entrypoint-initdb.d/db_data/ratings.csv' DELIMITER ',' CSV HEADER;

-- Insert data from the temporary table into the actual table
INSERT INTO ratings(timestamp, user_id, movie, rating)
SELECT tr.timestamp, tr.user_id, tr.movie, tr.rating
FROM temp_ratings tr
WHERE EXISTS (SELECT 1 FROM users u WHERE u.user_id = tr.user_id);

-- Drop the temporary table
DROP TABLE temp_ratings;
