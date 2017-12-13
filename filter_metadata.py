import pandas as pd

# Loads unique movie ids
ratings_df = pd.read_csv('ratings_small.csv', low_memory=False)
movie_ids = ratings_df.movieId.unique()

# Loads movie metadata and filter only by those which are in movie_ids
movies_df = pd.read_csv('movies_metadata.csv', low_memory=False)
movies_df['id'] = pd.to_numeric(movies_df['id'], downcast='integer')
small = movies_df[movies_df['id'].isin(list(movie_ids))]

# Saves in new file
small.to_csv('movies_metadata_small.csv', index=False)
