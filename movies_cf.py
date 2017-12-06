import pandas as pd

# Calculates the jaccard similarity of sets a and b
def jaccard_sim(a, b):
    intersection = a.intersection(b)
    union = a.union(b)
    return len(intersection) / len(union)

# Returns a DataFrame object of jaccard similarities
def jaccard_sim_find_users(userId, users):
    # Get the movies the user userId has seen
    a = users[users['userId'] == userId]
    a = set(a['movieId'].tolist())
    
    # Find the most similar users using jaccard similarity
    similarity_list = []
    for i in range(0, len(set(users['userId']))):
        b = users[users['userId'] == i]
        b = set(b['movieId'].tolist())
        similarity_list.append(jaccard_sim(a, b))
    
    # Convert the list to a DataFrame object
    similarity_df = pd.DataFrame({'similarity':similarity_list})
    # Add a column for userId
    similarity_df['userId'] = pd.DataFrame({'userId':range(0, len(set(users['userId'])))})
    # Sort it descending according to similarity
    similarity_df = similarity_df.sort_values(by=['similarity'], ascending=False)
    return similarity_df
    
# Returns a DataFrame object of movies that can be recommended to userId
def recommended_movies(userId, ratings_df):
    # First find the top5 (really top4) similar users according to jaccard sim
    similarity_df = jaccard_sim_find_users(userId, ratings_df)
    user_list = similarity_df['userId'].head().tolist()
    # Remove the rows and columns which are not needed
    movies_df = ratings_df.pivot(index='userId', columns='movieId', values='rating')
    movies_df = movies_df.loc[user_list]
    # Drop columns (movies) which have NaN values for other than userId
    movies_df = movies_df.dropna(axis=1, thresh=4, subset=[user_list[1:]])
    # Sort the movies so the NaN values are last
    movies_df = movies_df.sort_values(axis=1, by=user_list[0])
    return movies_df


ratings_df = pd.read_csv('ratings_small.csv')

# Bad example (no movies to recommend)
print("Movie list for userId 1:")
print(recommended_movies(1, ratings_df))

# Good example (5 movies to recommend)
print("Movie list for userId 3:")
print(recommended_movies(3, ratings_df))
# Correlation matrix of the good example
print(recommended_movies(3, ratings_df).T.corr(method='pearson'))
