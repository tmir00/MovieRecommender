# Movie Recommender Backend App

## **Introduction**

This backend application provides personalized movie recommendations through a simple API. Users can send a POST request with their user ID to receive a list of movie recommendations tailored to their preferences. If a generic user ID is provided, the app returns a list of generally popular movies.

### **Example Request**

```bash
curl -X POST http://localhost:5000/predict -H "Content-Type: application/json" -d '{"user_id": "<user_id>"}'
```
Replace <user_id> with the actual user's ID from the dataset for personalized recommendations, or use a generic ID for general popular recommendations.

### **Example Response**
```json
{
  "account": true,
  "movies": [
    "the+sword+in+the+stone+1963",
    "bettie+page+reveals+all+2013",
    "harry+potter+and+the+prisoner+of+azkaban+2004",
    "the+road+to+ruin+1934",
    "the+curse+of+the+were-rabbit+2005",
    "the+worlds+fastest+indian+2005",
    "in+which+we+serve+1942",
    "spirited+away+2001",
    "pulp+fiction+1994",
    "schindlers+list+1993"
  ]
}
```
The response includes a flag indicating whether the user has an account in our database ("account": true) and a list of the top 10 recommended movies.

<br>

## **Getting Started**
1. Clone the repository
```bash
git clone https://github.com/tmir00/MovieRecommender.git
```

2. Navigate to MovieRecommender directory:
```bash
cd MovieRecommender
```

3. Build and start the application using Docker:
``` bash
docker-compose up --build
```

4. You can now send POST requests to http://localhost:5000/predict to get movie recommendations. Use the curl command as shown above to test the endpoint.


<br>

## **Project Overview**

### **1. Data Preprocessing**
The movie recommendation model relies on a dataset collected from a Kafka stream during a semester project, which includes three CSV files:
- `movies.csv`: Contains data about movies, such as genres, production countries, production companies, original language, and overview.
- `users.csv`: Stores user demographic data, including age, gender, and occupation.
- `ratings.csv`: Records each user's rating of movies.

### **2. Database Integration**

The raw data collected is then integrated into a PostgreSQL database to facilitate efficient data manipulation and querying. This integration is accomplished using two SQL scripts:

- `01_schema.sql`: Defines the schema for the movies, users, and ratings tables.
- `02_import_data.sql`: Handles the importation of data from the CSV files into the corresponding tables in the PostgreSQL database.

The PostgreSQL database contains three primary tables to store movies, users, and their ratings. Below is an overview of these tables:

### List of Relations

The schema of the tables in the PostgreSQL database is as follows:
Schema | Name | Type | Owner
--------|---------|-------|-------
public | movies | table | user
public | ratings | table | user
public | users | table | user

### Movies Table:
#### Table: `public.movies`

| Column                | Type    | Collation | Nullable | Default |
|-----------------------|---------|-----------|----------|---------|
| id                    | text    |           | not null |         |
| tmdb_id               | text    |           |          |         |
| imdb_id               | text    |           |          |         |
| title                 | text    |           |          |         |
| original_title        | text    |           |          |         |
| adult                 | boolean |           |          |         |
| budget                | integer |           |          |         |
| genres                | text    |           |          |         |
| homepage              | text    |           |          |         |
| original_language     | text    |           |          |         |
| overview              | text    |           |          |         |
| popularity            | numeric |           |          |         |
| poster_path           | text    |           |          |         |
| production_companies  | text    |           |          |         |
| production_countries  | text    |           |          |         |
| release_date          | date    |           |          |         |
| revenue               | bigint  |           |          |         |
| runtime               | numeric |           |          |         |
| spoken_languages      | text    |           |          |         |
| status                | text    |           |          |         |
| vote_average          | numeric |           |          |         |
| vote_count            | numeric |           |          |         |

**Indexes:**
- `"movies_pkey" PRIMARY KEY, btree (id)`

### Users Table:
#### Table: `public.users`

| Column      | Type          | Collation | Nullable | Default |
|-------------|---------------|-----------|----------|---------|
| user_id     | integer       |           | not null |         |
| age         | integer       |           |          |         |
| occupation  | text          |           |          |         |
| gender      | character(1)  |           |          |         |

**Indexes:**
- `"users_pkey" PRIMARY KEY, btree (user_id)`

**Referenced by:**
- TABLE `"ratings"` CONSTRAINT `"ratings_user_id_fkey"` FOREIGN KEY (user_id) REFERENCES users(user_id)

### Ratings Table:
#### Table: `public.ratings`

| Column    | Type    | Collation | Nullable | Default |
|-----------|---------|-----------|----------|---------|
| timestamp | text    |           |          |         |
| user_id   | integer |           |          |         |
| movie     | text    |           |          |         |
| rating    | integer |           |          |         |

**Foreign-key constraints:**
- `"ratings_user_id_fkey"` FOREIGN KEY (user_id) REFERENCES users(user_id)

<br>

### **2. Machine Learning Model**

The core of the recommendation system is a machine learning model that processes the data to identify similar users and movies. The model works as follows:

1. **Collaborative Filtering**: Identifies users with similar attributes (age, gender, occupation) and aggregates their top-rated movies to recommend movies liked by similar users.

2. **Content-Based Filtering**: Analyzes movie features such as genres, production countries, and movie summaries to find movies similar to those the user has rated highly.

The model calculates cosine similarities between movies based on their features and stores these similarities for efficient retrieval during the recommendation process.

#### Key Functions:

- `find_highly_rated_movies_for_similar_users()`: Finds top-rated movies for users similar to the specified attributes.
- `find_similar_users()`: Identifies users similar to the given user based on age, gender, and occupation.
- `process_data()`: Preprocesses the movie data and calculates cosine similarities between movies.
- `train()`: Trains the model by processing data from the database and saving the computed features and similarities.

<br>

### **3. Flask Application**

The Flask application offers an API endpoint to handle recommendation requests. It performs error checking on the incoming requests and returns a response containing the recommended movies.

#### Example Request:

```bash
curl -X POST http://localhost:5000/predict -H "Content-Type: application/json" -d '{"user_id": "<user_id>"}'
```

<br>

## Process Flow Diagram
The diagram below summarizes and illustrates the data flow and operational processes within the project, from initial data processing to the final output of personalized movie recommendations.
![Process Flow](./processflow.png)

