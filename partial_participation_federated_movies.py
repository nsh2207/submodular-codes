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



if __name__ == "__main__":
    file_path = 'ratings.csv'
    ratings_df = pd.read_csv(file_path,sep='\t') # Read the CSV file into a pandas DataFrame
    tabular = ratings_df.pivot_table(index='user_id',columns='movie_id',values='rating').replace({np.NaN:0}) # Prepare the tabular data from the ratings data by pivoting the data on the movie_id, for each unique user_id and replacing the NaN values with 0, to prepare a clean table of users and movies with the ratings given by the users to the movies
    cumulativeRatingsSampled = {} # Keys are sample proportion and values are 2D list , where each sublist is the mean maximal ratings of the users in the tabular data for each subset size for the distributed client sampling algorithm at the current subsample proportion and seperate sublists are for running the algorithm again for the same subsample proportion
    # cumulativeRatingsSampled dictionary is to capture the randomness in the sampling of the users in the distributed client sampling algorithm. For each fixed k, we randomly choose clients. So we run the algorithm for 3 times to capture the mean and variance and show it as a band in the graph
    centralCumulativeRatings = [] # List to store the cumulative ratings of the users in the tabular data for each subset size for the centralized greedy algorithm
    cumulativeRatingsTotal = [] # List to store the cumulative ratings of the users in the tabular data for each subset size for the distributed full greedy algorithm
    subsetSizes = [size for size in range(5, 31, 5)] # List of subset sizes for which the maximal recommendation set is to be computed
    availableMovies = set(tabular.columns) # List of available movies in the tabular data for random selector function to randomly select a subset of movies
    randomlyGeneratedRatings = [] # List to store the cumulative ratings of the users in the tabular data for each subset size for the randomly selected subset of movies


    # ------------------- Uncomment the following block of code to run the distributed algorithm for various subsample proportions -------------------
    
    
    
    # userSize = tabular.shape[0]
    # sampleProportions = [0.001,0.01,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    # fig, axs = plt.subplots(len(sampleProportions), 1, figsize=(10, 5 * len(sampleProportions)))
    # for i, sampleProportion in enumerate(sampleProportions):
    #     subSet = []
    #     cumulativeRatings = []
    #     for subsetSize in subsetSizes:
    #         subSet = computeMaximalRecommendationSet(ratings_df, tabular, subsetSize=5, subSet=subSet, sampleProportion=sampleProportion)
    #         totalMaximalRatingsForAllClients = tabular[subSet].max(axis=1).sum()
    #         cumulativeRatings.append(totalMaximalRatingsForAllClients / userSize)
    #     axs[i].plot(subsetSizes, cumulativeRatings, marker='+', color='r', linestyle='None')
    #     axs[i].plot(subsetSizes, cumulativeRatings, color='b')
    #     axs[i].set_xlabel('Subset Size')
    #     axs[i].set_ylabel('Mean Maximal Ratings from each client')
    #     axs[i].set_title(f'Sample Proportion: {sampleProportion}')
    # plt.tight_layout()
    # plt.savefig('subsampled_cumulative_movielens.png')
    # plt.show()


    # ------------------- Uncomment the following block of code to run the distributed algorithm for various subsample proportions -------------------


##  Block of code to run the distributed algorithm for a subsample proportion of 10% and compare it with the centralized and distributed full greedy algorithms


    # Distributed Client Sampling of 10%, 5%, 1% running the algorithm 3 times, for each k, to capture the randomness in the sampling of the users in the distributed client sampling algorithm

    sampleProportions = [0.1,0.05,0.01] # 10% of the data is sampled
    
    userSize = tabular.shape[0] # Number of users in the tabular data
    for sampleProportion in sampleProportions:
        
        cumulativeRatingsSampled[sampleProportion] = [] # 2D List to store the cumulative ratings of the users in the tabular data for each subset size for the distributed client sampling algorithm at the current subsample proportion        
        print(f"Running for sample proportion {sampleProportion}")
        for t in range(3): # Run the algorithm 3 times to capture the randomness in the sampling of the users in the distributed client sampling algorithm
            subSet = [] # Maximum movie recommendation set till now
            print(f"Running for sample proportion {sampleProportion} and iteration {t}")
            cumulativeRatingsSampledForThisK = [] # List to store the cumulative ratings of the users in the tabular data for each subset size for the distributed client sampling algorithm at the current subsample proportion at this iteration
            for subsetSize in subsetSizes:
                # Though subsetSize is 5,10,15,20,... I am passing 5 as the subsetSize parameter to the function because
                # whole logic is incremental and the function will add 5 movies to the subset each time it is called and the
                # size of set collected till now from beginning is subsetSize

                subSet = computeMaximalRecommendationSet(tabular, subsetSize=5, subSet=subSet, sampleProportion=sampleProportion)
                totalMaximalRatingsForAllClients = tabular[subSet].max(axis=1).sum() # Total maximal ratings of the users in the tabular data for the subset of movies in the subSet
                meanMaximalRatings = totalMaximalRatingsForAllClients / userSize # Mean maximal ratings of the users in the tabular data for the subset of movies in the subSet
                cumulativeRatingsSampledForThisK.append(meanMaximalRatings) # Append the mean maximal ratings to the cumulativeRatingsSampled list
            cumulativeRatingsSampled[sampleProportion].append(cumulativeRatingsSampledForThisK) # Append the cumulativeRatingsSampledForThisK list to the cumulativeRatingsSampled list

   
   
    # Distributed Full Greedy
    
    sampleProportion = 1 # Full Greedy
    subSet = []
    for subsetSize in subsetSizes:
        # Though subsetSize is 5,10,15,20,... I am passing 5 as the subsetSize parameter to the function because
        # whole logic is incremental and the function will add 5 movies to the subset each time it is called and the
        # size of set collected till now from beginning is subsetSize

        subSet = computeMaximalRecommendationSet(tabular, subsetSize=5, subSet=subSet, sampleProportion=sampleProportion) # Compute the maximal recommendation set for the given subset size
        totalMaximalRatingsForAllClients = tabular[subSet].max(axis=1).sum()
        meanMaximalRatings = totalMaximalRatingsForAllClients / userSize
        cumulativeRatingsTotal.append(meanMaximalRatings)
    

 ## Centralized Greedy   
    movie_ids = tabular.columns # List of movie ids
    user_ids = tabular.index # List of user ids
    userwiseContributionToTheSum = [0]*len(user_ids) # Contribution of each user to the sum of the ratings of the users in the tabular data
    cumulativeRatingForAllMoviesPresentInTheSubsetTillnow = 0 # Sum of maximal ratings of the users in the tabular data for the subset of movies in the maximal recommendation set till now
    for subsetSize in subsetSizes:
        # Though subsetSize is 5,10,15,20,... I am passing 5 as the subsetSize parameter to the function because
        # whole logic is incremental and the function will add 5 movies to the subset each time it is called and the
        # size of set collected till now from beginning is subsetSize

        cumulativeRatingForThisSubsetSize,userwiseContributionToTheSum = computeMaximalRecommendationSetCentral(tabular, 5,userwiseContributionToTheSum) # The centralized greedy algorithm to compute the maximal recommendation set for the given subset size.
        # computeMaximalRecommendationSetCentral function returns each user's contribution to the maximal ratings from the subset of movies in the maximal recommendation set till now and the increase in sum of maximal ratings of the users due to currently included new subset of movies
        cumulativeRatingForAllMoviesPresentInTheSubsetTillnow += cumulativeRatingForThisSubsetSize # Update the sum of maximal ratings of the users in the tabular data for the subset of movies in the maximal recommendation set till now
        meanMaximalRatings = cumulativeRatingForAllMoviesPresentInTheSubsetTillnow/len(user_ids) # Mean maximal ratings of the users in the tabular data for the subset of movies in the maximal recommendation set till now
        centralCumulativeRatings.append(meanMaximalRatings)
    
    
    # Randomly generated subset of movies
    subSet = []
    for subsetSize in subsetSizes:
        # Though subsetSize is 5,10,15,20,... I am passing 5 as the subsetSize parameter to the function because
        # whole logic is incremental and the function will add 5 movies to the subset each time it is called and the
        # size of set collected till now from beginning is subsetSize
        subSet, meanMaximalRatings, availableMovies = randomSubsetGenerator(tabular, 5, subSet, availableMovies)
        randomlyGeneratedRatings.append(meanMaximalRatings)

    
    # Block of code to plot the graph of the mean ratings of the users in the tabular data for each subset size for the centralized greedy, distributed full greedy and distributed client sampling algorithms
    
    plt.plot(subsetSizes, centralCumulativeRatings, marker='*', label='Centralized', color='orange')
    plt.plot(subsetSizes, cumulativeRatingsTotal, marker='s', label='Distributed Full', color='blue')
    for sampleProportion in sampleProportions:
        meanRatings = np.mean(cumulativeRatingsSampled[sampleProportion], axis=0)
        stdRatings = np.std(cumulativeRatingsSampled[sampleProportion], axis=0)
        plt.plot(subsetSizes, meanRatings, marker='v', label=f'Client Sampling K= {int(sampleProportion*100)}%', linestyle='--')
        plt.fill_between(subsetSizes, meanRatings - stdRatings, meanRatings + stdRatings, alpha=0.2)
    plt.plot(subsetSizes, randomlyGeneratedRatings, marker='x', label='Randomly Selected Subset', color='green')
    plt.xlabel('Number of Movies')
    plt.ylabel('Function Value')
    plt.title('MovieLens')
    plt.legend()
    
    plt.savefig('combined_movie_updated.png')
    plt.show()
    print("#"*100)
    print("Data")
    print("Centralized Greedy")
    print(centralCumulativeRatings)
    print("Distributed Full Greedy")
    print(cumulativeRatingsTotal)
    print("Distributed Client Sampling")
    print(cumulativeRatingsSampled)
    

    
    