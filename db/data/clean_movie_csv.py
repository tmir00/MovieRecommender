import pandas as pd
import json


def json_to_pipe(json_str):
    """
    Convert a json formatted string into a string with attributes previously indexed by 'name' separated by pipe.

    :param json_str: String to be formatted.
    :return: A string with attributes separated by pipe.
    """
    try:
        if isinstance(json_str, str):
            # Replace single quotes with double quotes for valid JSON
            json_list = json.loads(json_str.replace("'", '"'))
            # Join the 'name' fields with a pipe
            return '|'.join([d['name'] for d in json_list if 'name' in d])
        else:
            return ''
    except json.decoder.JSONDecodeError as e:
        # If there's a JSON decode error, print the error and the offending string
        print(f"JSONDecodeError: {e}")
        print(f"Offending string: {json_str}")
        return ''


# Load the CSV file into a DataFrame
df = pd.read_csv('movies.csv')

# Remove JSON format to list of genres or production countries, etc.
df['genres'] = df['genres'].apply(json_to_pipe)
df['production_countries'] = df['production_countries'].apply(json_to_pipe)
df['production_companies'] = df['production_companies'].apply(json_to_pipe)
df['spoken_languages'] = df['spoken_languages'].apply(json_to_pipe)

# Convert budget and revenue to integers
df['budget'] = df['budget'].fillna(-1)
df['revenue'] = df['revenue'].fillna(-1)

# Drop unnecessary columns
df.drop('belongs_to_collection', axis=1, inplace=True)

# Save the cleaned DataFrame back to CSV
df.to_csv('movies.csv', index=False)
