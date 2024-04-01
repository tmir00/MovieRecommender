import csv
import json
import os
import unittest

import numpy as np
import pandas as pd
import unittest
from unittest.mock import patch, Mock
import requests

from request_utils import get_movie, USER_URL
from trainModel import extract_genre_names, extract_country_names, preprocessing_demographics, \
    pre_processing_content_based_filtering, run_cosine_computation, save_results_to_file
from model import run_model, demographic_filtering

from request_utils import get_user


class TestExtractGenreNames(unittest.TestCase):
    def test_single_genre(self):
        genres_list = [{'id': 18, 'name': 'Drama'}]
        self.assertEqual(extract_genre_names(genres_list), 'Drama')

    def test_multiple_genres(self):
        genres_list = [{'id': 18, 'name': 'Drama'}, {'id': 10749, 'name': 'Romance'}]
        self.assertEqual(extract_genre_names(genres_list), 'Drama,Romance')

    def test_empty_list(self):
        genres_list = []
        self.assertEqual(extract_genre_names(genres_list), '')

    def test_missing_name_key(self):
        genres_list = [{'id': 18}]
        self.assertEqual(extract_genre_names(genres_list), '')


class TestExtractCountryNames(unittest.TestCase):
    def test_single_country(self):
        country_list = [{'iso_3166_1': 'FR', 'name': 'France'}]
        self.assertEqual(extract_country_names(country_list), 'France')

    def test_multiple_countries(self):
        country_list = [{'iso_3166_1': 'FR', 'name': 'France'}, {'iso_3166_1': 'IT', 'name': 'Italy'}]
        self.assertEqual(extract_country_names(country_list), 'France,Italy')

    def test_empty_list(self):
        country_list = []
        self.assertEqual(extract_country_names(country_list), '')

    def test_missing_name_key(self):
        country_list = [{'iso_3166_1': 'FR'}]
        self.assertEqual(extract_country_names(country_list), '')

    class TestPreProcessingContentBasedFiltering(unittest.TestCase):

        def setUp(self):
            # Sample test data
            self.test_data = pd.DataFrame({
                'id': ['burn+up+1991', 'brighton+beach+memoirs+1986', 'the+crimson+wing+mystery+of+the+flamingos+2008',
                       'holy+motors+2012'],
                'genres': ["[{'id': 16, 'name': 'Animation'}]",
                           "[{'id': 35, 'name': 'Comedy'}, {'id': 18, 'name': 'Drama'}, {'id': 10749, 'name': 'Romance'}]",
                           "[{'id': 99, 'name': 'Documentary'}]",
                           "[{'id': 18, 'name': 'Drama'}, {'id': 14, 'name': 'Fantasy'}]"],
                'overview': ['overview1', 'overview2', 'overview3', 'overview4'],
                'production_countries': ["[{'iso_3166_1': 'JP', 'name': 'Japan'}]",
                                         "[{'iso_3166_1': 'US', 'name': 'United States of America'}]",
                                         "[{'iso_3166_1': 'GB', 'name': 'United Kingdom'}, {'iso_3166_1': 'US', 'name': 'United States of America'}]",
                                         "[{'iso_3166_1': 'FR', 'name': 'France'}]"],
                'original_language': ['ja', 'en', 'en', 'fr'],
                'vote_average': [5, 6.2, 7.3, 7],
                'vote_count': [2, 20, 6, 232],
                'title': ['Burn Up!', 'Brighton Beach Memoirs', 'The Crimson Wing: Mystery of the Flamingos',
                          'Holy Motors']
            })

            # Expected processed data
            self.expected_data = pd.DataFrame({
                'id': ['burn+up+1991', 'brighton+beach+memoirs+1986', 'the+crimson+wing+mystery+of+the+flamingos+2008',
                       'holy+motors+2012'],
                'genres': ["[{'id': 16, 'name': 'Animation'}]",
                           "[{'id': 35, 'name': 'Comedy'}, {'id': 18, 'name': 'Drama'}, {'id': 10749, 'name': 'Romance'}]",
                           "[{'id': 99, 'name': 'Documentary'}]",
                           "[{'id': 18, 'name': 'Drama'}, {'id': 14, 'name': 'Fantasy'}]"],
                'overview': ['overview1', 'overview2', 'overview3', 'overview4'],
                'production_countries': ['Japan', 'United States of America', 'United Kingdom,United States of America',
                                         'France'],
                'original_language': ['ja', 'en', 'en', 'fr'],
                'vote_average': [5, 6.2, 7.3, 7],
                'vote_count': [2, 20, 6, 232],
                'title': ['Burn Up!', 'Brighton Beach Memoirs', 'The Crimson Wing: Mystery of the Flamingos',
                          'Holy Motors'],
                'genre_names': ['Animation', 'Comedy,Drama,Romance', 'Documentary', 'Drama,Fantasy']
            })

        def test_preProcessingContentBasedFiltering(self):
            pre_processing_content_based_filtering(self.test_data)
            processed_data = pd.read_csv('processed_movie_details.csv')

            # Assert that the processed data matches the expected data
            pd.testing.assert_frame_equal(processed_data, self.expected_data)

        def tearDown(self):
            # Clean up the CSV file after the test
            os.remove('processed_movie_details.csv')


class TestComputeAndSaveCosineSimilarities(unittest.TestCase):

    def setUp(self):
        # Sample test data
        self.test_data = pd.DataFrame({
            'id': ['movie1', 'movie2', 'movie3'],
            'genres': ["[{'id': 1, 'name': 'Action'}]", "[{'id': 1, 'name': 'Action'}]",
                       "[{'id': 2, 'name': 'Comedy'}]"],
            'overview': ['This is an action movie.', 'This is also an action movie.', 'This is a comedy movie.'],
            'production_countries': ["['United States of America']", "['United States of America']", "['France']"],
            'original_language': ['en', 'en', 'fr'],
            'vote_average': [7.5, 7.5, 6.5],
            'vote_count': [100, 100, 200],
            'title': ['Action Movie 1', 'Action Movie 2', 'Comedy Movie'],
            'genre_names': ['Action', 'Action', 'Comedy']
        })

    def test_compute_and_save_cosine_similarities(self):
        valid_n_pca_components = min(self.test_data.shape[0] - 1, self.test_data.shape[1] - 1, 100)
        movie_similarity_results = run_cosine_computation(self.test_data, n=2, n_pca_components=valid_n_pca_components)

        # # Call the function with test data
        # compute_and_save_cosine_similarities(self.test_data)

        # Check if the dictionary has been populated
        self.assertTrue(movie_similarity_results)

        # Check for the presence of specific keys and their types
        for movie_id, results in movie_similarity_results.items():
            self.assertIn('similar_movies', results)
            self.assertIn('scores', results)
            self.assertIsInstance(results['similar_movies'], list)
            self.assertIsInstance(results['scores'], list)

            # Check if the number of similar movies and scores match the expected 'n=2' value
            self.assertEqual(len(results['similar_movies']), 2)
            self.assertEqual(len(results['scores']), 2)

    def tearDown(self):
        # Clean up the file after the test
        if os.path.exists("movie_similarity_results.json"):
            os.remove("movie_similarity_results.json")


class TestPreprocessingDemographics(unittest.TestCase):

    def setUp(self):
        # Sample test data for ratings_df
        self.ratings_data = pd.DataFrame({
            'Timestamp': ['2023-09-24T12:43:12', '2023-09-24T12:43:13', '2023-09-24T12:43:13', '2023-09-24T12:43:17'],
            'Id': [245338, 195922, 289337, 309863],
            'Status': ['GET', 'GET', 'GET', 'GET'],
            'Type': ['rate', 'rate', 'rate', 'rate'],
            'movie_id': ['my+little+eye+2002', 'licence+to+kill+1989', 'the+princess+diaries+2+royal+engagement+2004',
                      'harry+potter+and+the+order+of+the+phoenix+2007'],
            'rating': [4, 5, 3, 4]
        })

        # Sample test data for user_details_df
        self.user_details_data = pd.DataFrame({
            'user_id': [245338, 195922, 289337, 309863],
            'age': [26, 26, 51, 28],
            'occupation': ['college/grad student', 'college/grad student', 'academic/educator', 'technician/engineer'],
            'gender': ['M', 'M', 'M', 'M'],
            'message': ['', '', '', '']
        })

    def test_preprocessingDemographics(self):
        preprocessing_demographics(self.ratings_data, self.user_details_data, 'processed_demographics.csv')

        # Load the processed data
        processed_data = pd.read_csv('processed_demographics.csv')

        # Expected processed data
        expected_data = pd.DataFrame({
            'user_id': [245338, 195922, 289337, 309863],
            'age': [26, 26, 51, 28],
            'occupation': ['college/grad student', 'college/grad student', 'academic/educator', 'technician/engineer'],
            'gender': ['M', 'M', 'M', 'M'],
            'Timestamp': ['2023-09-24T12:43:12', '2023-09-24T12:43:13', '2023-09-24T12:43:13', '2023-09-24T12:43:17'],
            'Status': ['GET', 'GET', 'GET', 'GET'],
            'Type': ['rate', 'rate', 'rate', 'rate'],
            'movie_id': ['my+little+eye+2002', 'licence+to+kill+1989', 'the+princess+diaries+2+royal+engagement+2004',
                      'harry+potter+and+the+order+of+the+phoenix+2007'],
            'rating': [4, 5, 3, 4]
        })

        # Assert that the processed data matches the expected data
        pd.testing.assert_frame_equal(processed_data, expected_data)

    def tearDown(self):
        # Clean up the processed CSV file after the test
        os.remove('processed_demographics.csv')


class TestDemographicFiltering(unittest.TestCase):

    def setUp(self):
        # Sample test data for merged_df
        self.merged_data = pd.DataFrame({
            'user_id': [1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
            'age': [26, 26, 51, 51, 28, 28, 30, 30, 30, 30],
            'occupation': ['college/grad student', 'college/grad student', 'academic/educator', 'academic/educator', 'technician/engineer', 'technician/engineer', 'other', 'other', 'other', 'other'],
            'gender': ['M', 'M', 'M', 'M', 'M', 'M', 'F', 'F', 'F', 'F'],
            'movie_id': ['A', 'B', 'A', 'C', 'B', 'C', 'A', 'D', 'D', 'E'],
            'rating': [4, 4, 3, 4, 4, 3, 4, 5, 4, 2]
        })

    def test_demographicFiltering(self):
        # Test for a user in the dataset
        top_movie_for_user_1 = demographic_filtering(1, self.merged_data)
        self.assertEqual(top_movie_for_user_1, 'A')

        # Test for a user not in the dataset but with demographic data similar to user 4 and 5
        top_movie_for_user_999999 = demographic_filtering(999999, self.merged_data, n=1, age_weight=0.5, gender_weight=0.3, occupation_weight=0.2,
                                                          threshold=0.5)
        self.assertEqual(top_movie_for_user_999999, 'A')  # This should be 'A' based on the test data


class TestRunModel(unittest.TestCase):

    def setUp(self) -> None:
        user_details_df = pd.read_csv('./data_v1/user_details_output_26_09_23.csv')
        ratings_df = pd.read_csv('./data_v1/sample_data_24_09_23_rating.csv')
        preprocessing_demographics(ratings_df, user_details_df, 'processed_demographics.csv')
        movie_details_df = pd.read_csv('./data_v1/movie_details_output_26_09_23.csv')
        pre_processing_content_based_filtering(movie_details_df, 'processed_movie_details.csv')
        temp_movie_details = pd.read_csv('processed_movie_details.csv')
        movie_similarity_results_put = run_cosine_computation(temp_movie_details)
        save_results_to_file(movie_similarity_results_put, 'movie_similarity_results.json')

    def test_runModel_no_errors(self):

        # Test with various user_ids
        user_ids = [1, 100, 500, 1000, 5000]
        for user_id in user_ids:
            try:
                run_model(user_id)
            except Exception as e:
                self.fail(f"runModel raised an exception for user_id {user_id}: {e}")


class TestRequestHandler(unittest.TestCase):

    def setUp(self):
        self.user_data = [
            {"user_id": 1, "age": 34, "occupation": "sales/marketing", "gender": "M"},
            {"user_id": 2, "age": 33, "occupation": "college/grad student", "gender": "M"},
            {"user_id": 3, "age": 29, "occupation": "scientist", "gender": "M"}
        ]

    @patch('request_utils.requests.get')
    def test_get_user_successful(self, mock_get):
        for user in self.user_data:
            with self.subTest(user=user):
                mock_response = Mock()
                mock_response.json.return_value = user
                mock_get.return_value = mock_response

                result = get_user(user['user_id'])
                mock_get.assert_called_with(USER_URL + str(user['user_id']), timeout=100)
                self.assertEqual(result, user)

    @patch('request_utils.requests.get')
    def test_get_user_nonexistent(self, mock_get):
        mock_response = Mock()
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError
        mock_get.return_value = mock_response

        with self.assertRaises(requests.exceptions.HTTPError):
            get_user(23847327432847328423847328974238947239847328472)

    @patch('request_utils.requests.get')
    def test_get_user_timeout(self, mock_get):
        user_id = 1
        mock_get.side_effect = requests.exceptions.Timeout

        with self.assertRaises(requests.exceptions.Timeout):
            get_user(user_id)

    @patch('request_utils.requests.get')
    def test_get_user_http_error(self, mock_get):
        user_id = 1
        mock_response = Mock()
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError
        mock_get.return_value = mock_response

        with self.assertRaises(requests.exceptions.HTTPError):
            get_user(user_id)

    class TestGetMovie(unittest.TestCase):

        @patch('request_utils.requests.get')
        def test_get_movie_success(self, mock_get):
            """Test successful retrieval of a movie."""
            movie_name = "get+shorty+1995"
            mock_response = Mock()
            mock_response.json.return_value = {
                "id": movie_name,
                "tmdb_id": 8012,
                "imdb_id": "tt0113161",
                "title": "Get Shorty",
                "original_title": "Get Shorty",
                "adult": "False",
                "belongs_to_collection": {"id": 91698, "name": "Chili Palmer Collection",
                                          "poster_path": "/ae3smJDdWrMJ77tDpYOrpo4frKq.jpg",
                                          "backdrop_path": "/uWaANGQeoSs5vSP1CWtlkDrkqei.jpg"},
                "budget": "30250000",
                "genres": [
                    {"id": 35, "name": "Comedy"},
                    {"id": 53, "name": "Thriller"},
                    {"id": 80, "name": "Crime"}
                ],
                "homepage": "null",
                "original_language": "en",
                "overview": "Chili Palmer is a Miami mobster who gets sent by his boss, the psychopathic \"Bones\" Barboni, to collect a bad debt from Harry Zimm, a Hollywood producer who specializes in cheesy horror films. When Chili meets Harry's leading lady, the romantic sparks fly. After pitching his own life story as a movie idea, Chili learns that being a mobster and being a Hollywood producer really aren't all that different.",
                "popularity": "12.669608",
                "poster_path": "/vWtDUUgQAsVyvRW4mE75LBgVm2e.jpg",
                "production_companies": [
                    {"name": "Jersey Films", "id": 216},
                    {"name": "Metro-Goldwyn-Mayer (MGM)", "id": 8411}
                ],
                "production_countries": [
                    {"iso_3166_1": "US", "name": "United States of America"}
                ],
                "release_date": "1995-10-20",
                "revenue": "115101622",
                "runtime": 105,
                "spoken_languages": [
                    {"iso_639_1": "en", "name": "English"}
                ],
                "status": "Released",
                "vote_average": "6.4",
                "vote_count": "305"
            }
            mock_response.raise_for_status.return_value = None
            mock_get.return_value = mock_response

            result = get_movie(movie_name)
            self.assertEqual(result['title'], "Get Shorty")
            self.assertEqual(result['imdb_id'], "tt0113161")
            self.assertEqual(result['adult'], "False")
            self.assertEqual(result['genres'][0]['name'], "Comedy")
            self.assertEqual(result['budget'], "30250000")
            self.assertEqual(result['original_language'], "en")
            self.assertEqual(result['overview'].startswith("Chili Palmer is a Miami mobster"), True)
            self.assertEqual(result['production_countries'][0]['name'], "United States of America")
            self.assertEqual(result['release_date'], "1995-10-20")
            self.assertEqual(result['runtime'], 105)
            self.assertEqual(result['status'], "Released")
            self.assertEqual(result['vote_average'], "6.4")

        @patch('request_utils.requests.get')
        def test_get_movie_http_error(self, mock_get):
            """Test retrieval of a movie with an HTTP error."""
            movie_name = "unknown_movie"
            mock_response = Mock()
            mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError
            mock_get.return_value = mock_response

            with self.assertRaises(requests.exceptions.HTTPError):
                get_movie(movie_name)

        @patch('request_utils.requests.get')
        def test_get_movie_timeout(self, mock_get):
            """Test retrieval of a movie with a timeout error."""
            movie_name = "get+shorty+1995"
            mock_get.side_effect = requests.exceptions.Timeout

            with self.assertRaises(requests.exceptions.Timeout):
                get_movie(movie_name)

class TestGetMovie(unittest.TestCase):
    @patch('request_utils.requests.get')
    def test_get_movie_success(self, mock_get):
        """Test successful retrieval of a movie."""
        movie_name = "get+shorty+1995"
        mock_response = Mock()
        mock_response.json.return_value = {
            "id": movie_name,
            "tmdb_id": 8012,
            "imdb_id": "tt0113161",
            "title": "Get Shorty",
            "original_title": "Get Shorty",
            "adult": "False",
            "belongs_to_collection": {"id": 91698, "name": "Chili Palmer Collection", "poster_path": "/ae3smJDdWrMJ77tDpYOrpo4frKq.jpg", "backdrop_path": "/uWaANGQeoSs5vSP1CWtlkDrkqei.jpg"},
            "budget": "30250000",
            "genres": [
                {"id": 35, "name": "Comedy"},
                {"id": 53, "name": "Thriller"},
                {"id": 80, "name": "Crime"}
            ],
            "homepage": "null",
            "original_language": "en",
            "overview": "Chili Palmer is a Miami mobster who gets sent by his boss, the psychopathic \"Bones\" Barboni, to collect a bad debt from Harry Zimm, a Hollywood producer who specializes in cheesy horror films. When Chili meets Harry's leading lady, the romantic sparks fly. After pitching his own life story as a movie idea, Chili learns that being a mobster and being a Hollywood producer really aren't all that different.",
            "popularity": "12.669608",
            "poster_path": "/vWtDUUgQAsVyvRW4mE75LBgVm2e.jpg",
            "production_companies": [
                {"name": "Jersey Films", "id": 216},
                {"name": "Metro-Goldwyn-Mayer (MGM)", "id": 8411}
            ],
            "production_countries": [
                {"iso_3166_1": "US", "name": "United States of America"}
            ],
            "release_date": "1995-10-20",
            "revenue": "115101622",
            "runtime": 105,
            "spoken_languages": [
                {"iso_639_1": "en", "name": "English"}
            ],
            "status": "Released",
            "vote_average": "6.4",
            "vote_count": "305"
        }
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        result = get_movie(movie_name)
        self.assertEqual(result['title'], "Get Shorty")
        self.assertEqual(result['imdb_id'], "tt0113161")
        self.assertEqual(result['adult'], "False")
        self.assertEqual(result['genres'][0]['name'], "Comedy")
        self.assertEqual(result['budget'], "30250000")
        self.assertEqual(result['original_language'], "en")
        self.assertEqual(result['overview'].startswith("Chili Palmer is a Miami mobster"), True)
        self.assertEqual(result['production_countries'][0]['name'], "United States of America")
        self.assertEqual(result['release_date'], "1995-10-20")
        self.assertEqual(result['runtime'], 105)
        self.assertEqual(result['status'], "Released")
        self.assertEqual(result['vote_average'], "6.4")

    @patch('request_utils.requests.get')
    def test_get_movie_http_error(self, mock_get):
        """Test retrieval of a movie with an HTTP error."""
        movie_name = "unknown_movie"
        mock_response = Mock()
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError
        mock_get.return_value = mock_response

        with self.assertRaises(requests.exceptions.HTTPError):
            get_movie(movie_name)

    @patch('request_utils.requests.get')
    def test_get_movie_timeout(self, mock_get):
        """Test retrieval of a movie with a timeout error."""
        movie_name = "get+shorty+1995"
        mock_get.side_effect = requests.exceptions.Timeout

        with self.assertRaises(requests.exceptions.Timeout):
            get_movie(movie_name)


if __name__ == '__main__':
    unittest.main()

