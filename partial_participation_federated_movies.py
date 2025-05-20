import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
import warnings
from centralized_greedy_movies import computeMaximalRecommendationSetCentral
warnings.filterwarnings("ignore")



def computeMaximalRecommendationSet(tabular,subsetSize,subSet,sampleProportion):
    '''
    Function to compute the maximal recommendation set for a given subset size
    The function takes the tabular data, the subset size, the subset of movies collected in the maximal set till now, and the sample proportion (k) as input
    and returns the subset of movies that maximizes the sum of the ratings of the users in the tabular data till now
    '''
    movie_ids = tabular.columns # Get the movie ids from the tabular data
    subsetCount = subsetSize # Number of movies to be included in the subset
    
    for t in range(subsetCount):
        tabular_sample = tabular.sample(frac=sampleProportion) # selecting a random sample of the tabular data
        
        # Block of code to compute the maximum rating of each user in the tabular_sample data, among the subset of movies included in the subSet till now

        if len(subSet) > 0:
            tabularSampleSubset = tabular_sample[subSet]
            tabularSampleSubset['max'] = tabularSampleSubset.max(axis=1)
        else:
            tabularSampleSubset = pd.DataFrame([0]*tabular_sample.shape[0],columns=['max'])
            tabularSampleSubset.index = tabular_sample.index
        
        
        # Block of code to find the next new movie to be included in the subset such that the sum of the ratings of the users in the tabular_sample data is maximized

        nextMovieToBeIncluded = -1 # next movie to be included in the subset
        nextMovieMaxContribution = -1 # maximum contribution of the next movie to be included in the subset
        
        # This for loop iterates over all the movies in the tabular data and computes the contribution of each movie to the sum of the ratings of the users in the tabular_sample data

        for movie in movie_ids:
            currentMovieContribution = np.array(tabular_sample[movie]) # Contribution of the current movie to the sum of the ratings of the users in the tabular_sample data. Finding the new contribution the current movie can bring out from each user in the tabular_sample data.
            currentMovieContribution = np.maximum(currentMovieContribution - tabularSampleSubset['max'], 0) # If it doesn't increase the maximum rating of the user, it is set to 0 else it is the difference between the rating of the movie and the maximum rating of the user
            thisMovieContribution = np.sum(currentMovieContribution) # Sum of contributions of the current movie from all the users in the tabular_sample data. This is the local greedy criteria and we aim to maximize this quantity for every new inclusion of a movie in the subset 
            if thisMovieContribution > nextMovieMaxContribution:
                nextMovieMaxContribution = thisMovieContribution
                nextMovieToBeIncluded = movie
        subSet.append(nextMovieToBeIncluded) # Append the next movie to be included in the subset to the subSet list
        
    return subSet # Return the subset of movies that maximizes the sum of the ratings of the users in the tabular data till now


def randomSubsetGenerator(tabular, subsetSize, subSet, availableMovies):
    '''
        Function to randomly select a unique subset of movies of the given size from the tabular data
        The function takes the tabular data and the subset size as input and returns the randomly selected unique subset of movies
        '''
    for i in range(subsetSize):
        movie = np.random.choice(list(availableMovies)) # Randomly select a movie from the available movies
        subSet.append(movie) # Append the selected movie to the subSet list
        availableMovies.discard(movie) # Remove the selected movie from the available movies
    meanMaximalRating = tabular[subSet].max(axis=1).sum()/tabular.shape[0] # Compute the mean of the maximal ratings of the users in the tabular data for the subset of movies in the subSet
    return subSet, meanMaximalRating, availableMovies # Return the randomly selected unique subset of movies



