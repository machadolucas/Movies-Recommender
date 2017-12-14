import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


#
# Compute a similarity matrix between an unseen movie and a list of seen movies 
#
# INPUT: id of unseen movie, list of seen movies, base matrix for similarity matrix
# OUTPUT: similarity matrix sorted by similarity to movieId
#
def cos_sim_matrix(movieId, movies_user, base_matrix):
    # Insert movie movieId to the list so we can calculate its cosine similarity to other movies
    movies_user.insert(0, movieId)
    similarity_matrix = base_matrix
    similarity_matrix = similarity_matrix.loc[movies_user]
    similarity_matrix = similarity_matrix.fillna(0)
    similarity_matrix = pd.DataFrame(cosine_similarity(similarity_matrix), index=movies_user, columns=movies_user)
    return similarity_matrix.sort_values(by=movieId, ascending=False, axis=1)
    # Alternative way
    # similarity_matrix = pd.DataFrame(cosine_similarity(similarity_matrix.loc[[movieId]], similarity_matrix), index=[movieId], columns=movies_user)
    # return similarity_matrix.sort_values(by=movieId, ascending=False, axis=1)


#
# Compute a predicted rating for an unseen movie
#
# INPUT: id of unseen movie, id of user, similarity matrix, number of neighbors, ratings file
# OUTPUT: a float value of the predicted rating
#
def predicted_rating(movieId, userId, sim_matrix, neighbors, ratings_df):
    # Get most similar movieIds and similarities
    movies = sim_matrix.columns.values.tolist()[1:neighbors + 1]
    sim_values = sim_matrix.T[movieId].values.tolist()[1:neighbors + 1]

    # Calculate prediction using the formula on page 22 of CF slides
    top = 0.0
    for i in range(0, neighbors):
        # userId's rating for the current movie
        rating = ratings_df[ratings_df['userId'] == userId]
        rating = float(rating[rating['movieId'] == movies[i]]['rating'])
        # Multiply it with similarity of movieId and the current movie
        top += sim_values[i] * rating
    bot = 0.0
    for i in range(0, neighbors):
        bot += sim_values[i]
    # Final rating
    if bot != 0:
        return top / bot
    return 0


#
# Item-based collaborative filtering
# Basic idea:
# 1. Find movies the user hasn't seen
# 2. Compare unseen movies to seen movies and find the most similar ones
# 3. Use the ratings of similar seen movies to predict the rating of unseen movies
#
# INPUT: id of user, number of neighbors to consider, ratings file
# OUTPUT: DataFrame object of predicted ratings sorted by highest rating
#
def item_based_cf_ratings(userId, neighbors, ratings_df):
    # Movies the user has seen
    movies_user = set(ratings_df[ratings_df['userId'] == userId]['movieId'].tolist())
    # All movies
    movies_all = set(ratings_df['movieId'])
    # Unseen movies by the user userId
    movies_unseen = list(movies_all.difference(movies_user))

    # Base for similarity matrix
    base_matrix = ratings_df.pivot(index='movieId', columns='userId', values='rating')

    # Calculate a predicted rating for each movie the user hasn't seen
    predicted_ratings = []
    for i in range(0, len(movies_unseen)):
        movieId = movies_unseen[i]
        sim_matrix = cos_sim_matrix(movieId, list(movies_user), base_matrix)
        rating = predicted_rating(movieId, userId, sim_matrix, neighbors, ratings_df)
        predicted_ratings.append(rating)
        # Calculating ratings takes a while so it's good to have some feedback
        print("i=" + str(i) + " movieId=" + str(movies_unseen[i]) + " rating=" + str(rating))
    # Make a DataFrame object of the ratings and sort it by rating
    df = pd.DataFrame({'movieId': movies_unseen[:len(predicted_ratings)], 'pred_rating': predicted_ratings})
    return df.sort_values(by='pred_rating', ascending=False)
