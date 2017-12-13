import pandas as pd
import numpy as np
from ast import literal_eval
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet

ratings_df = pd.read_csv('ratings_small.csv')
movies_df = pd.read_csv('movies_metadata.csv', low_memory=False)

# Converts json of genres to list of strings
movies_df['genres'] = movies_df['genres'].fillna('[]').apply(literal_eval).apply(
    lambda x: [i['name'] for i in x] if isinstance(x, list) else [])

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
    ['title', 'year', 'vote_count', 'vote_average', 'popularity', 'genres']]
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
