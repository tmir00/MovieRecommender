import pandas as pd

# Load the CSV file into a DataFrame
df = pd.read_csv('users.csv')

# Remove rows where 'user_id' is NaN
df = df.dropna(subset=['user_id'])

# Convert 'user_id' to integers
df['user_id'] = df['user_id'].astype(float).astype(int)
df['age'] = df['age'].astype(float).astype(int)

# Drop duplicates based on the 'user_id' column
df.drop_duplicates(subset='user_id', keep='first', inplace=True)

# Save the cleaned DataFrame back into the same CSV file
df.to_csv('users.csv', index=False)
