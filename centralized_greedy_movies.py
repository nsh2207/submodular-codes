import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
# Read the CSV file into a pandas DataFrame

# Display the first few rows of the DataFrame

def computeMaximalRecommendationSetCentral(tabular,subsetSize,userwiseContributionToTheSum):
    '''
    This function takes the the tabular representation of the ratings dataframe, the subset size, and the userwiseContributionToTheSum as input
    (The userwiseContributionToTheSum is the contribution of the user to the sum of the maximal ratings of the movies in the subset collected till now)
    and returns the maximal set of movies to be recommended such that the sum of the ratings of the movies in the subset is maximized
    '''
    movie_ids = tabular.columns # List of movie ids
    user_ids = tabular.index# List of user ids
    recommendationSumOfTheIncludedMovies = 0 # Variable to store the sum of the changes in the sum of ratings of the movies in the subset brought by the newly added movies
    subsetCount = subsetSize # Number of movies to be included in the subset
    for t in range(subsetCount):
        nextMovieToBeIncluded = -1 # Variable to store the next movie to be included in the subset
        nextMovieMaxContribution = -1 # Variable to store the maximum contribution of the next movie to be included in the subset
        for movie in movie_ids:
            currentMovieContribution = np.array(tabular[movie]) # Contribution of the current movie to the sum of the ratings of the movies in the subset
            currentMovieContribution = np.maximum(currentMovieContribution - userwiseContributionToTheSum, 0) # If for a user, the newly added movie increases the contribution to the sum of the ratings of the movies in the subset, then consider the positive difference, else keep 0
            thisMovieContribution = np.sum(currentMovieContribution)# Sum of contributions from all users. This is the local greedy criteria being considered at each step locally. This should be maximum for the next movie to be included in the subset 
            if thisMovieContribution > nextMovieMaxContribution:
                nextMovieMaxContribution = thisMovieContribution 
                nextMovieToBeIncluded = movie
        userwiseContributionToTheSum = np.maximum(userwiseContributionToTheSum, np.array(tabular[nextMovieToBeIncluded])) # Update the userwiseContributionToTheSum by considering the newly added movie
        recommendationSumOfTheIncludedMovies += nextMovieMaxContribution # update the sum of changes in the sum of ratings of the movies in the subset brought by the newly added movie by considering the maximum contribution of the newly added movie
        print(recommendationSumOfTheIncludedMovies/len(user_ids))
    
    return recommendationSumOfTheIncludedMovies,userwiseContributionToTheSum # Return the total change the newly added movies bring to the sum of the ratings of the movies in the subset and the userwiseContributionToTheSum



if __name__ == "__main__":
    subsetSizes = [size for size in range(5, 31, 5)] # List of subset sizes
    cumulativeRatings = [] # List to store the mean maximal ratings of the movies in the subset for all users
    file_path = 'ratings.csv'
    ratings_df = pd.read_csv(file_path,sep='\t')# Read the CSV file into a pandas DataFrame
    tabular = ratings_df.pivot_table(index='user_id',columns='movie_id',values='rating').replace({np.NaN:0})# Convert the ratings dataframe into a tabular representation where rows are users and columns are movies and the values are the ratings given by the users to the movies
    movie_ids = tabular.columns # List of movie ids
    user_ids = tabular.index # List of user ids
    userwiseContributionToTheSum = np.array([0]*len(user_ids)) # Initialize the userwiseContributionToTheSum to 0 for all users. It is the contribution of each user to the sum of the maximal ratings of the movies in the subset collected till now
    cumulativeRatingForAllMoviesPresentInTheSubsetTillnow = 0 # Total sum of the maximal ratings of the movies in the subset collected till now from all users
    
    
    # for subsetSize in subsetSizes:
    #     # Though subsetSize is 5,10,15,20,... I am passing 5 as the subsetSize parameter to the function because
    #     # whole logic is incremental and the function will add 5 movies to the subset each time it is called and the
    #     # size of set collected till now from beginning is subsetSize

    #     cumulativeRatingForThisSubsetSize,userwiseContributionToTheSum = computeMaximalRecommendationSetCentral(tabular, 5,userwiseContributionToTheSum) # 
    #     cumulativeRatingForAllMoviesPresentInTheSubsetTillnow += cumulativeRatingForThisSubsetSize # Update the total sum of the maximal ratings of the movies in the subset collected till now from all users with the change brought by the newly added 5 movies
    #     meanMaximalRatings = cumulativeRatingForAllMoviesPresentInTheSubsetTillnow/len(user_ids)# Compute the mean maximal ratings of the movies in the subset for all users
    #     cumulativeRatings.append(meanMaximalRatings)# Append the mean maximal ratings of the movies in the subset for all users to the cumulativeRatings list
    
    # Block to plot the graph of the mean maximal ratings of the movies in the subset for all users for each subset size
    cumulativeRatingForThisSubsetSize,userwiseContributionToTheSum = computeMaximalRecommendationSetCentral(tabular,200,userwiseContributionToTheSum) 
    
    # plt.xlabel('Subset Size')
    # plt.ylabel('Mean Maximal Ratings from each client')
    # plt.title('Maximal set of movies to be recommended globally')
    # plt.plot(subsetSizes, cumulativeRatings, marker='+', color='r', linestyle='None')
    # plt.plot(subsetSizes, cumulativeRatings, color='b')
    # plt.savefig('centralized_mean_movielens.png')
    # plt.show()

    
    

        