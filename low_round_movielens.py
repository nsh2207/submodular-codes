import sys
import matplotlib.pyplot as plt
import random
import numpy as np
from collections import defaultdict
from itertools import product
import pandas as pd
import pdb

def tupleGenerate(row, d,a_r_prime_array,X,S):
    """
    row : row of the tabular data (each client)
    d : number of tuples to be selected
    a_r_prime_array : array of movies with fixed order
    X : array of movies
    S : set of movies selected till now
    """
    
    r_prime = len(a_r_prime_array)
    cartesian_product = np.transpose([np.tile(X, r_prime), np.repeat(np.arange(r_prime), len(X))])
    # Select tuples
    selected_tuples = cartesian_product[np.random.choice(cartesian_product.shape[0], d, replace=False)]
 
    
    tupleStore = {}
    for selected_tuple in selected_tuples:
        e = selected_tuple[0]
        j = selected_tuple[1]        
        marginal_movie_list = a_r_prime_array[0:j+1].copy()
        marginal_movie_list = np.append(marginal_movie_list,e)
        if len(S) != 0:
            marginal_movie_list = np.append(marginal_movie_list,np.array(list(S)))
        before_movie_list = a_r_prime_array[0:j+1].copy()
        if len(S) != 0:
            before_movie_list = np.append(before_movie_list,np.array(list(S)))
        
        tupleStore[tuple(selected_tuple)] = row[marginal_movie_list].max() - row[before_movie_list].max()
    
    tupleStorePandas = pd.DataFrame(tupleStore.items(), columns=['tuple', 'Marginal Contribution'])
    # pdb.set_trace() 
    
    return tupleStorePandas


def lowRoundSubmodularImplement(K,d_cap,d_ratio,r,matrix,epsilon,tou,log_file):
    """
    K : fraction of clients to be selected
    d_ratio : proportion of tuples to be selected
    d_cap : maximum number of tuples to be selected
    r : size of the subset
    matrix : tabular data
    epsilon : parameter for updating the threshold 
    tou : threshold
    log_file : log file to store the output
    """
    S = set()
    i = 0
    while len(S) < r:        
        movies = np.array(matrix.columns)
        X = movies
        # print("#Outer Loop Iteration: ",i)
        while len(X)>0:
            r_prime = r-len(S)
            
            sampledClients = matrix.sample(frac=K) # selecting K% of the clients or rows from matrix data
            a_r_prime_array = np.random.choice(X,r_prime) # Randomly choose r_prime movies from the movies and fixing the order
            # print(a_r_prime_array)
            d_decided = min(d_cap,int(d_ratio*len(X)*r_prime)) # deciding the number of tuples to be selected
            print(d_decided,file=log_file,flush=True)
            sampledClients['tupleStore'] = sampledClients.apply(lambda row: tupleGenerate(row,d_decided,a_r_prime_array,X,S),axis=1) # for each active client getting marginal contributions of d uniformly randomly selected tuples from Xx{0,1,2,...,r_prime-1}
            central_tuple_store =  pd.concat(sampledClients['tupleStore'].tolist()).groupby('tuple').sum() # summing up the marginal contributions of the same tuple across all the clients  # Reducing by key
            central_tuple_store = central_tuple_store.reset_index()
            central_tuple_store[['j','e']] = central_tuple_store['tuple'].apply(lambda x: pd.Series([x[1], x[0]])) # splitting the tuple into j and e
            central_tuple_store = central_tuple_store[['j','e','Marginal Contribution']]
            central_tuple_store['Marginal Contribution'] = central_tuple_store['Marginal Contribution']*((len(X)*r_prime)/(d_decided*sampledClients.shape[0]))
            central_tuple_store_exceeded_threshold = central_tuple_store[central_tuple_store['Marginal Contribution'] >= tou] # filtering the tuples whose marginal contribution is greater than tou
            central_tuple_store_exceeded_threshold = central_tuple_store_exceeded_threshold[central_tuple_store_exceeded_threshold[['j','e']].apply(lambda x: True if len(S.union(set(a_r_prime_array[0:x['j']+1]).union({x['e']})))<=r+1 else False,axis=1)] # filtering the tuples whose union of S, a_r_prime_array[0:j+1] and {e} is less than or equal to r
            if central_tuple_store_exceeded_threshold.empty:
                print("Central Tuple Store Exceeded Threshold is empty",file=log_file,flush=True)
                break
            central_tuple_store_exceeded_threshold = central_tuple_store_exceeded_threshold.groupby('j').agg({'e': lambda x: list(set(x))}).reset_index() # grouping the tuples by j and getting unique e values for each j such that the marginal contribution is greater than tou and the union of S, a_r_prime_array[0:j+1] and {e} is less than or equal to r
            central_tuple_store_exceeded_threshold = central_tuple_store_exceeded_threshold[central_tuple_store_exceeded_threshold['e'].apply(lambda x: True if len(x)<=(1-epsilon)*len(X) else False)] # filtering the j values whose count of unique values of e is less than or equal to (1-epsilon)*|X|
            if central_tuple_store_exceeded_threshold.empty:
                print("Central Tuple Store Exceeded Threshold is empty",file=log_file,flush=True)
                break
            central_tuple_store_exceeded_threshold.sort_values('j',inplace=True) # sorting by j
            central_tuple_store_exceeded_threshold = central_tuple_store_exceeded_threshold.reset_index()
            X = central_tuple_store_exceeded_threshold.iloc[0]['e'] # updating X with the unique values of e for the minimum j value
            S = S.union(a_r_prime_array[0:central_tuple_store_exceeded_threshold.iloc[0]['j']+1]) # updating S with the first j values of a_r_prime_array
            print(S,file=log_file,flush=True)
            print(tabular[list(S)].max(axis=1).mean(),file=log_file,flush=True)
        
        tou = (1-epsilon)*tou # updating tou after X is empty or S is full
        print(tou,file=log_file,flush=True)
        if len(S) >= r:
            break
        i += 1

    return S

if __name__ == "__main__":

    subsetSizes = [size for size in range(5, 31, 5)] # List of subset sizes
    file_path = 'ratings.csv'
    ratings_df = pd.read_csv(file_path,sep='\t') # Read the CSV file into a pandas DataFrame
    tabular = ratings_df.pivot_table(index='user_id',columns='movie_id',values='rating').replace({np.NaN:0}) # Convert the ratings dataframe into a tabular representation where rows are users and columns are movies and the values are the ratings given by the users to the movies
    d_cap = 200
    d_ratio = 0.7
    k_ratio = 1.0
    log_file = "d={}k={}.txt".format(d_cap,k_ratio)
    with open(log_file, 'a') as log_file_p:
        print(log_file,file=log_file_p,flush=True)
        print(d_cap,d_ratio,k_ratio,file=log_file_p,flush=True)
        moviesSub = lowRoundSubmodularImplement(k_ratio,d_cap,d_ratio,200,tabular,0.405,0.006,log_file=log_file_p)
        print("######################",file=log_file_p,flush=True)
        print(moviesSub,file=log_file_p,flush=True)
        print(tabular[list(moviesSub)].max(axis=1).mean(),file=log_file_p,flush=True)

            


