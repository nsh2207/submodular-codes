import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from collections import defaultdict
from partial_participation_federated_movies import computeMaximalRecommendationSet
from centralized_greedy_movies import computeMaximalRecommendationSetCentral
from multiprocessing import Pool

def dict_to_dataframe(dictionary):
    return pd.DataFrame(dictionary.items(), columns=['movie_id', 'change'])


def accumulateChangeBoughtByMovies(row, moviesRemaining, movieConsiderProportion):
    changeBoughtByMovies = defaultdict(int)
    movies_to_consider = np.random.choice(list(moviesRemaining), size=int(len(moviesRemaining) * movieConsiderProportion), replace=False)
    for movie in movies_to_consider:
        changeBoughtByMovies[movie] += max(0, row[movie] - row['max'])
    return changeBoughtByMovies

def computeMaximalRecommendationSetLowBit(tabular,subsetSize,subSet,sampleProportion,movieConsiderProportion):
    movie_ids = tabular.columns # Get the movie ids from the tabular data
    subsetCount = subsetSize # Number of movies to be included in the subset
    # `moviesRemaining` is a set that keeps track of the movies that have not been included in the
    # subset yet. It is initialized with all the movie IDs available in the tabular data and is
    # updated throughout the process as movies are chosen to be included in the subset.
    moviesRemaining = set(movie_ids).difference(set(subSet)) # List of movies remaining to be included in the subset
    changeBoughtByMovies = defaultdict(int)
    for t in range(subsetCount):
        tabular_sample = tabular.sample(frac=sampleProportion) # selecting a random sample of the tabular data
        
        if len(subSet) == 0:
            tabular_sample['max'] = 0
        else:
            tabular_sample['max'] = tabular_sample[subSet].max(axis=1) # Maximum rating of the user in the tabular_sample data
        
        tabular_sample['changes_dictionary'] = tabular_sample.apply(lambda row: accumulateChangeBoughtByMovies(row, moviesRemaining, movieConsiderProportion), axis=1)
        combined_changes = pd.concat(tabular_sample['changes_dictionary'].apply(dict_to_dataframe).tolist()).groupby('movie_id').sum()
        nextMovieToBeIncluded = combined_changes.idxmax() # Find the next new movie to be included in the subset such that the sum of the ratings of the users in the tabular_sample data is maximized
        # Block of code to find the next new movie to be included in the subset such that the sum of the ratings of the users in the tabular_sample data is maximized
        subSet.append(nextMovieToBeIncluded.iloc[0]) # Append the next movie to be included in the subset to the subSet list
        moviesRemaining.discard(nextMovieToBeIncluded.iloc[0]) # Remove the next movie to be included in the subset from the movies remaining to be included in the subset
        
    return subSet # Return the subset of movies that maximizes the sum of the ratings of the users in the tabular data till now

if __name__ == "__main__":
    subsetSizes = [size for size in range(5, 31, 5)] # List of subset sizes
    file_path = 'ratings.csv'
    ratings_df = pd.read_csv(file_path,sep='\t') # Read the CSV file into a pandas DataFrame
    tabular = ratings_df.pivot_table(index='user_id',columns='movie_id',values='rating').replace({np.NaN:0}) # Convert the ratings dataframe into a tabular representation where rows are users and columns are movies and the values are the ratings given by the users to the movies
    userCount = tabular.shape[0] # Number of users in the tabular data
    # subSet = [] # List to store the subset of movies that maximizes the sum of the ratings of the users in the tabular data
    # sampleProportion = 0.3 # Proportion of the tabular data to be sampled
    # movieConsiderProportion = 0.4
    # meanRatings = [] # List to store the mean maximal ratings of the movies in the subset for all users
    # centralizedMeanRatings = [] # List to store the mean maximal ratings of the movies in the subset for all users for centralized algorithm
    # distributedMeanRatings = [] # List to store the mean maximal ratings of the movies in the subset for all users for distributed subsampled algorithm
    
    
    
    # for subsetSize in subsetSizes:
    #     subSet = computeMaximalRecommendationSetLowBit(tabular, 5,subSet,sampleProportion,movieConsiderProportion) # Compute the subset of movies that maximizes the sum of the ratings of the users in the tabular data till now
    #     mean = tabular[subSet].max(axis=1).sum()/userCount # Mean Maximum rating of the user in the tabular data
    #     meanRatings.append(mean)
    #     print(subSet) # Print the subset of movies that maximizes the sum of the ratings of the users in the tabular data till now

    # subSet = [] # Initialize the subset of venues to be empty initially for distributed subsampled algorithm
    # for subsetSize in subsetSizes:
    #     subSet = computeMaximalRecommendationSet(tabular, 5,subSet,sampleProportion)
    #     mean = tabular[subSet].max(axis=1).sum()/userCount
    #     distributedMeanRatings.append(mean)
    
    
#     user_ids = tabular.index # List of user ids
#     userwiseContributionToTheSum = np.array([0]*len(user_ids)) # Initialize the userwiseContributionToTheSum to 0 for all users. It is the contribution of each user to the sum of the maximal ratings of the movies in the subset collected till now
#     cumulativeRatingForAllMoviesPresentInTheSubsetTillnow = 0 # Total sum of the maximal ratings of the movies in the subset collected till now from all users
#     for subsetSize in subsetSizes:
#         cumulativeRatingForThisSubsetSize,userwiseContributionToTheSum = computeMaximalRecommendationSetCentral(tabular, 5,userwiseContributionToTheSum)
#         cumulativeRatingForAllMoviesPresentInTheSubsetTillnow += cumulativeRatingForThisSubsetSize
#         meanMaximalRatings = cumulativeRatingForAllMoviesPresentInTheSubsetTillnow/len(user_ids)
#         centralizedMeanRatings.append(meanMaximalRatings)

    


# labelForLowBit = 'Subsampling Distributed Low bit k={}% and d={}%'.format(100*sampleProportion, 100*movieConsiderProportion)
# labelDistributed = 'Subsampling Distributed k={}%'.format(100*sampleProportion)
# plt.plot(subsetSizes,meanRatings,marker='o', label=labelForLowBit, color='orange') # Plot the graph of the mean maximal ratings of the movies in the subset for all users for each subset size
# plt.plot(subsetSizes,centralizedMeanRatings,marker='o', label='Centralized version', color='blue') # Plot the graph of the mean maximal ratings of the movies in the subset for all users for each subset size
# plt.plot(subsetSizes,distributedMeanRatings,marker='o', label=labelDistributed, color='green') # Plot the graph of the mean maximal ratings of the movies in the subset for all users for each subset size
# plt.xlabel('Subset Size')
# plt.ylabel('Mean Maximal Ratings from each client')
# plt.title('Mean Maximal Ratings from each client vs Subset Size')
# plt.legend() # Show the legend
# plt.savefig('distributed_low_bit_movies_d_0.4.png') # Save the plot as a png file
# plt.show() # Show the plot

subSampleProportions = [0.01,0.1,0.3,0.5] # List of sub-sample proportions
movieConsiderProportions = [0.1,0.2,0.4,0.6] # List of movie consider proportions
fig, axs = plt.subplots(len(subSampleProportions), len(movieConsiderProportions), figsize=(20, 20))
fig.subplots_adjust(hspace=0.5, wspace=0.5)

# Precompute the distributedMeanRatings for each sampleProportion
distributedMeanRatings_dict = {}
for sampleProportion in subSampleProportions:
    print(f'Computing distributedMeanRatings for sampleProportion={sampleProportion}')
    distributedMeanRatings = []
    subSet = []
    for subsetSize in subsetSizes:
        subSet = computeMaximalRecommendationSet(tabular, subsetSize, subSet, sampleProportion)
        mean = tabular[subSet].max(axis=1).sum() / userCount
        distributedMeanRatings.append(mean)
    distributedMeanRatings_dict[sampleProportion] = distributedMeanRatings

for i, sampleProportion in enumerate(subSampleProportions):
    for j, movieConsiderProportion in enumerate(movieConsiderProportions):
        meanRatings = []
        subSet = []
        print(f'Sample: {sampleProportion*100}%, Movie: {movieConsiderProportion*100}%')
        for subsetSize in subsetSizes:
            subSet = computeMaximalRecommendationSetLowBit(tabular, subsetSize, subSet, sampleProportion, movieConsiderProportion)
            mean = tabular[subSet].max(axis=1).sum() / userCount
            meanRatings.append(mean)
        
        axs[i, j].plot(subsetSizes, meanRatings, marker='o', label='Low Bit', color='orange')
        axs[i, j].plot(subsetSizes, distributedMeanRatings_dict[sampleProportion], marker='o', label='Distributed', color='green')
        axs[i, j].set_title(f'Sample: {sampleProportion*100}%, Movie: {movieConsiderProportion*100}%')
        axs[i, j].set_xlabel('Subset Size')
        axs[i, j].set_ylabel('Mean Maximal Ratings')
        axs[i, j].legend()

plt.suptitle('Mean Maximal Ratings vs Subset Size for Different Sample and Movie Consider Proportions')
plt.savefig('grid_subplots_various_k_and_d_.png')
plt.show()
