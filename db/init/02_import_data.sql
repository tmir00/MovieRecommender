COPY movies(movie_id, title, genres) FROM '/docker-entrypoint-initdb.d/db_data/movies.csv' WITH (FORMAT csv, HEADER true);
COPY ratings(user_id, movie_id, rating, timestamp) FROM '/docker-entrypoint-initdb.d/db_data/ratings.csv' WITH (FORMAT csv, HEADER true);
COPY tags(user_id, movie_id, tag, timestamp) FROM '/docker-entrypoint-initdb.d/db_data/tags.csv' WITH (FORMAT csv, HEADER true);
COPY links(movie_id, imdb_id, tmdb_id) FROM '/docker-entrypoint-initdb.d/db_data/links.csv' WITH (FORMAT csv, HEADER true);
