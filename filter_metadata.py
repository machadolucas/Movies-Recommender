import pandas as pd

# Loads unique movie ids
ratings_df = pd.read_csv('ratings_small.csv', low_memory=False)
movie_ids = ratings_df.movieId.unique()

# Converts to tmdbIds
links_df = pd.read_csv('links_small.csv', low_memory=False)
tmdb_ids = links_df[links_df['movieId'].isin(list(movie_ids))].tmdbId.values

# Loads movie metadata and filter only by those which are in movie_ids
movies_df = pd.read_csv('movies_metadata.csv', low_memory=False)
movies_df['id'] = pd.to_numeric(movies_df['id'], downcast='integer')
small = movies_df[movies_df['id'].isin(list(tmdb_ids))]

# Saves in new file
small.to_csv('movies_metadata_small.csv', index=False)
