import pandas as pd
import numpy as np
from ast import literal_eval
import movies_cf
import sys
from collections import Counter

ratings_df = pd.read_csv('ratings_small.csv')
movies_df = pd.read_csv('movies_metadata_small.csv', low_memory=False)
links_df = pd.read_csv('links_small.csv')

# Converts json of genres to list of strings
movies_df['genres'] = movies_df['genres'].fillna('[]').apply(literal_eval).apply(
    lambda x: [i['name'] for i in x] if isinstance(x, list) else [])

# Return a list of genres sorted by their frequency in movies seen by the user
# (ties are sorted alphabetically)
def find_user_genres(userId, normalized):
    # Movies the user has seen
    movies_user = ratings_df[ratings_df['userId'] == userId]['movieId'].tolist()
    # Convert the movies from grouplens-movieid to tmdb-movieid
    tmdb_movies = list(links_df[links_df['movieId'].isin(movies_user)].tmdbId.values)
    # Find out the genres of those movies
    genre_list = []
    genre_dict = {}
    for movie in tmdb_movies:
        genres = movies_df[movies_df['id'] == movie]['genres']
        # Convert tmdb-movieid to grouplens-movieid to get the rating
        mId = int(links_df[links_df['tmdbId'] == movie]['movieId'])
        rating = float(ratings_df[(ratings_df['userId'] == userId) & (ratings_df['movieId'] == mId)]['rating'])

        # Some movies don't have genres 
        if genres.empty:
            continue
        # but most do
        genres = genres.tolist()[0]
        for genre in genres:
            genre_list.append(genre)
            genre_dict.setdefault(genre, 0.0)
            genre_dict[genre] += (2 * rating) / 10
    # Normalized ratings
    if (normalized):
        return sorted(Counter(genre_dict).most_common(), key=lambda x: (-x[1], x[0]))
    # Non-normalized ratings
    else:
        return sorted(Counter(genre_list).most_common(), key=lambda x: (-x[1], x[0]))


# Calculate the genre score of movieId for userId
# Genre score is a float value between 0 and 1
def genre_score(user_genres, total_genres, movieId):
    # Convert grouplens-movieid to tmdb-movieid
    movieId = int(links_df[links_df['movieId'] == movieId]['tmdbId'])
    # List of genres of movieId
    movie_genres = movies_df[movies_df['id'] == movieId]['genres']
    if movie_genres.empty:
        return 0
    else:
        movie_genres = movie_genres.tolist()[0]
        genre_score = 0
        for genre in movie_genres:
            if genre in user_genres:
                genre_score += user_genres[genre]
        return float(genre_score) / total_genres


'''
Uses IMDB's weighted rating to define which movies are relevant to be recommended.

The basic idea behind this recommender is that movies that are more popular and more critically acclaimed will have
 a higher probability of being liked by the average audience. This model does not give personalized recommendations
  based on the user (yet!).
  
v is the number of votes for the movie
m is the minimum votes required to be listed in the chart
R is the average rating of the movie
C is the mean vote across the whole report

 To determine an appropriate value for m, the minimum votes required to be listed in the chart. We use 85th percentile
  as our cutoff. In other words, for a movie to feature in the charts, it must have more votes than at least 85% of
   the movies in the list.
'''

vote_counts = movies_df[movies_df['vote_count'].notnull()]['vote_count'].astype('int')
vote_averages = movies_df[movies_df['vote_average'].notnull()]['vote_average'].astype('int')
C = vote_averages.mean()
m = vote_counts.quantile(0.85)

# Generalizes release date to year
movies_df['year'] = pd.to_datetime(movies_df['release_date'], errors='coerce').apply(
    lambda x: str(x).split('-')[0] if x != np.nan else np.nan)
# Mounts the list of qualified movies to recommend
qualified = movies_df[
    (movies_df['vote_count'] >= m) & (movies_df['vote_count'].notnull()) & (movies_df['vote_average'].notnull())][
    ['title', 'id', 'year', 'vote_count', 'vote_average', 'popularity', 'genres']]
qualified['vote_count'] = qualified['vote_count'].astype('int')
qualified['vote_average'] = qualified['vote_average'].astype('int')

'''
Therefore, to qualify to be considered for the chart, a movie has to have at least 82 (m) votes on TMDB. We also see
 that the average rating for a movie on TMDB is 5.244 (C). 6832 (qualified.shape[0]) Movies qualify to be on our chart.
'''


# IMDB's weighted rating
def weighted_rating(x):
    v = x['vote_count']
    R = x['vote_average']
    return (v / (v + m) * R) + (m / (m + v) * C)


# Adds column with weighted rating and sort by it
qualified['wr'] = qualified.apply(weighted_rating, axis=1)
qualified = qualified.sort_values('wr', ascending=False)

# List of qualified movies
qualified_movies_tmdb = list(qualified.id.values)
# Convert tmdbIds to grouplensIds
qualified_movies_id = list(links_df[links_df['tmdbId'].isin(qualified_movies_tmdb)].movieId.values)

# Use only the qualified movies to calculate rating-based recommendations
ratings_df = ratings_df[ratings_df['movieId'].isin(qualified_movies_id)]

# User to do predictions for
userId = int(sys.argv[1])
# Desired neighborhood size
neighbors = 10
# Because we filtered the ratings file we might not have enough neighbors so we'll use what we got
user_movie_count = len(set(ratings_df[ratings_df['userId'] == userId]['movieId'].tolist()))
if user_movie_count < neighbors:
    neighbors = user_movie_count


print("Calculating item-based recommendations...")
cf_movies = movies_cf.item_based_cf_ratings(userId, neighbors, ratings_df)
#print(list(asdf['movieId']))

# List of genres of the movies userId has seen
user_genres = dict(find_user_genres(userId, True))
total_genres = sum(dict(find_user_genres(userId, False)).values())

# Loop through the recommendations
final_ratings = []
final_movieIds = []
final_tmdbIds = []
final_titles = []
for movie in list(cf_movies['movieId']):
    print("Calculating final score for movieId " + str(movie))
    final_rating = 0.0
    # Change the scale from 0-5 to 0-10 and add the cf score
    final_rating += float(2 * cf_movies[cf_movies['movieId'] == movie]['pred_rating'])
    # Add the popularity-based rating
    tmdbId = links_df[links_df['movieId'] == movie]['tmdbId']
    final_rating += float(qualified[qualified['id'] == int(tmdbId)]['wr'])
    # Get the average of the two
    final_rating = final_rating / 2
    # Add up the genre score
    final_rating += float(genre_score(user_genres, total_genres, movie))
    # ROFLmao
    final_titles.append(str(qualified[qualified['id'] == int(tmdbId)].title.values))
    final_movieIds.append(int(movie))
    final_tmdbIds.append(int(tmdbId))
    final_ratings.append(final_rating)

# Make a DataFrame object of the ratings and sort it by rating
#final_recommendations = pd.DataFrame({'movieId': final_movieIds, 'tmdbId': final_tmdbIds, 'score': final_ratings})
final_recommendations = pd.DataFrame({'Title': final_titles, 'score': final_ratings})
final_recommendations = final_recommendations.sort_values(by='score', ascending=False)
final_recommendations = final_recommendations.reset_index(drop=True)
final_recommendations.index = final_recommendations.index + 1
print("Recommendations for userId " + str(userId) + " using neighborhood size " + str(neighbors) + ":")
print(final_recommendations.head(25))
