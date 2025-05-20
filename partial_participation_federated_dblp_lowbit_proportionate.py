from indexed_pq import IndexedMaxPQ
import sys
import matplotlib.pyplot as plt
import random
from centralized_greedy import compute_greedy_maximal_set_central
import numpy as np
from collections import defaultdict
from partial_participation_federated_dblp import compute_greedy_maximal_set
def read_graph(file_path):
    '''
    Read the graph from the file and return the graph as a dictionary with keys as authors and values as the set of venues
    '''
    graph = {}
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split(':')
            node = parts[0].strip()
            neighbors = [author.strip() for author in parts[1][1:-1].split(',')]
            graph[node] = set(neighbors)
    return graph


def transpose_graph_generator(graph):
    '''
    Takes the graph as input and returns the transpose of the graph i.e. the graph with the edges reversed i.e. 
    the keys are the venues and the values are the set of authors covered by the corresponding venue
    '''
    transpose_graph = {}
    for node, neighbors in graph.items():
        for neighbor in neighbors:
            if neighbor not in transpose_graph:
                transpose_graph[neighbor] = set()
            transpose_graph[neighbor].add(node)
    return transpose_graph



def subSampleGraphGenerator(graph,subsetProportion):
    '''
    Takes the graph and the subset proportion as input and returns a sub sampled graph with the subset proportion of the original graph.
    It takes the dictionary with authors as keys and the set of venues as values and returns a dictionary with randomly selected subset of authors as keys and the set of their corresponding venues as values.
    For example, if the subset proportion is 0.1, then 10% of the authors are randomly selected and the corresponding venues are included in the sub sampled graph.
    '''

    sample_size = int(subsetProportion*len(graph))

    # Randomly sample items
    sampled_items = random.sample(list(graph.items()), sample_size)

    # Convert sampled items back to a dictionary (if needed)
    subSampledGraph = dict(sampled_items)
    return subSampledGraph


def computeGreedyMaximalSetLowBit(graph, transposeGraph, subsetSize,venuesLeft, subSet,subSampleProportion,venueConsiderProportion):
    
    for t in range(subsetSize):
        subGraph = subSampleGraphGenerator(graph,subSampleProportion) # Generate the sub sampled graph
        authorsAlreadyCovered = []
        for authors,venues in subGraph.items():
            if venues.intersection(subSet):
                authorsAlreadyCovered.append(authors)
        
        for author in authorsAlreadyCovered:
            subGraph.pop(author)
        
        differenceBoughtByVenue = defaultdict(int)

        for author,venues in subGraph.items():
            nextChosenVenuesByClient = np.random.choice(venuesLeft,size=int(venueConsiderProportion*venuesLeft.shape[0]),replace=False)
            for nextChosenVenueByClient in nextChosenVenuesByClient:
                if nextChosenVenueByClient in venues:
                    differenceBoughtByVenue[nextChosenVenueByClient]+=1
                else:
                    differenceBoughtByVenue[nextChosenVenueByClient]+=0
                # differenceBoughtByVenue[nextChosenVenueByClient]+=1
            
        nextVenueToBeIncluded = max(differenceBoughtByVenue,key=differenceBoughtByVenue.get)
        venuesLeft = np.delete(venuesLeft, np.where(venuesLeft == nextVenueToBeIncluded))
        subSet.append(nextVenueToBeIncluded)
        print(nextVenueToBeIncluded)
    
    coveredAuthors = set()
    for venue in subSet:
        coveredAuthors = coveredAuthors.union(transposeGraph[venue])
    return subSet,len(coveredAuthors),venuesLeft

if __name__ == "__main__":
    file_path = 'graph.txt'
    print("Reading the graph...")
    subsetSizes = [size for size in range(5, 31, 5)]
    # authorsCoveredCountsSampled dictionary captures the randomness of the sub-sampling process and the algorithm by running the algorithm multiple times for each sub-sample proportion
    authorsCoveredCountsLowBitTotal = {} # Percentage of authors covered in 100% authors algorithm for each subset size
    authorsCoveredCountsTotalDistributed = {} # Percentage of authors covered in distributed algorithm (non lowbit) for each subset size
    authorsCoveredCountsCentral = [] # Percentage of authors covered in centralized algorithm for each subset size
    graph = read_graph(file_path) # Generate the graph from the file
    transpose_graph = transpose_graph_generator(graph) # Generate the transpose graph from the graph
    totalAuthors = len(graph) # Total number of authors in the graph
    subSampleProportions = [0.1] # List of sub-sample proportions
    venueConsiderProportions = [0.01,0.1,0.2,0.4,0.9]
    
    for subSampleProportion in subSampleProportions: 
        print(f"subSampleProportion:{subSampleProportion} full bit")
        authorsCoveredCountsTotalDistributed[subSampleProportion] = [] # List to store the percentage of authors covered by the subset of venues for each subset size for the current sub-sample proportion
        for i in range(3): # Run the distributed subsampled algorithm 3 times
            subSet = [] # Initialize the subset of venues to be empty initially for distributed subsampled algorithm
            authorsCoveredCountsTotalDistributedThisTime = [] # List to store the cumulative ratings of the users in the tabular data for each subset size for the distributed client sampling algorithm at the current subsample proportion at this iteration
            for subsetSize in subsetSizes: 
                # Though subsetSize is 5,10,15,20,... I am passing 5 as the subsetSize parameter to the function because 
                # whole logic is incremental and the function will add 5 venues to the subset each time it is called and the 
                # size of set collected till now from beginning is subsetSize
                        
                subSet, authorsCovered = compute_greedy_maximal_set(graph, transpose_graph, subsetSize=5, subSet=subSet, subSampleProportion=subSampleProportion)
                authorsCoveredCountsTotalDistributedThisTime.append(100*authorsCovered/len(graph))

            authorsCoveredCountsTotalDistributed[subSampleProportion].append(authorsCoveredCountsTotalDistributedThisTime)

        for venueConsiderProportion in venueConsiderProportions:
            if venueConsiderProportion == 0.9 and subSampleProportion >0.01:
                continue
            print(f"subSampleProportion:{subSampleProportion},venueConsiderProportion:{venueConsiderProportion}") 
            authorsCoveredCountsLowBitTotal[(subSampleProportion,venueConsiderProportion)] = [] # List to store the percentage of authors covered by the subset of venues for each subset size for the current sub-sample proportion
            for i in range(3):                
                print("Iteration: ",i)
                subSet =  [] # List to store the subset of venues that maximize the number of authors covered
                authorsCoveredCountsLowBitTotalThisTime = [] # List to store the percentage of authors covered by the subset of venues for each subset size for the current sub-sample proportion for this iteration
                venuesLeft = set(transpose_graph.keys()).difference(subSet)
                venuesLeft = np.array(list(venuesLeft))
                for subsetSize in subsetSizes:                
                    subSet,coveredAuthorsCount,venuesLeft =computeGreedyMaximalSetLowBit(graph,transpose_graph,5,subSet=subSet,subSampleProportion=subSampleProportion,venuesLeft=venuesLeft,venueConsiderProportion=venueConsiderProportion)
                    authorsCoveredCountsLowBitTotalThisTime.append(100*coveredAuthorsCount/totalAuthors)
                    print(subSet)
                
                authorsCoveredCountsLowBitTotal[(subSampleProportion,venueConsiderProportion)].append(authorsCoveredCountsLowBitTotalThisTime)


            


    authorsCovered = 0 # Number of authors covered by the top venues till now
    for subsetSize in subsetSizes:
        # Though subsetSize is 5,10,15,20,... I am passing 5 as the subsetSize parameter to the function because 
        # whole logic is incremental and the function will add 5 venues to the subset each time it is called and the 
        # size of set collected till now from beginning is subsetSize

        venuesList = compute_greedy_maximal_set_central(graph,transpose_graph, subsetSize=5)
        for venue in venuesList:
            authorsCovered += venue[1] # Add the number of authors covered by the top venue to the total number of authors covered till now
        authorsCoveredCountsCentral.append(100*authorsCovered/totalAuthors) # Appending percentage of authors covered by the top venues for the current subset size to the list
   
    print("#######################################################")
    print("data produced in the algorithm")
    print("Centralized")
    print(authorsCoveredCountsCentral)
    print("Low Bit")
    print(authorsCoveredCountsLowBitTotal)
    print("Distributed")
    print(authorsCoveredCountsTotalDistributed)

        
      
    plt.plot(subsetSizes,authorsCoveredCountsCentral,marker='o', label='Centralized', color='blue') # Plot the graph of the percentage of authors covered by the subset of venues for each subset size
    colors = ['orange','green','red','purple','brown','pink','gray','olive','cyan','magenta']
    for i,sampleProportion in enumerate(subSampleProportions):
        row = subSampleProportions.index(sampleProportion)
        distributedMeanRatings = np.mean(authorsCoveredCountsTotalDistributed[sampleProportion], axis=0)
        distributedStdRatings = np.std(authorsCoveredCountsTotalDistributed[sampleProportion], axis=0)
        plt.plot(subsetSizes, authorsCoveredCountsCentral, label='Centralized Greedy',color='black',marker='o')
        plt.plot(subsetSizes, distributedMeanRatings, label=f'Distributed Full Bit Sample: {sampleProportion*100}%',color = colors[i],marker='o')
        plt.fill_between(subsetSizes, distributedMeanRatings - distributedStdRatings, distributedMeanRatings + distributedStdRatings, alpha=0.1,color = colors[i])
        
        for j,movieConsiderProportion in enumerate(venueConsiderProportions):
            if movieConsiderProportion == 0.9 and sampleProportion >0.01:
                continue
            lowBitRatings = authorsCoveredCountsLowBitTotal[(sampleProportion, movieConsiderProportion)]
            meanLowBitRatings = np.mean(lowBitRatings, axis=0)
            stdLowBitRatings = np.std(lowBitRatings, axis=0)
            
            plt.plot(subsetSizes, meanLowBitRatings, label=f'Movie: {movieConsiderProportion*100}%',color = colors[(i+j+1)%11],marker='o')   
            plt.fill_between(subsetSizes, meanLowBitRatings - stdLowBitRatings, meanLowBitRatings + stdLowBitRatings, alpha=0.1,color = colors[(i+j+1)%11])
            
            
        plt.title(f'Client Sampling K= {sampleProportion*100}%')
        plt.xlabel('Number of Movies')
        plt.ylabel('Function Value')
        plt.legend()
        plt.savefig(f'algorithm_3_graphs/DBLP_varying_d_at_fixed_k={sampleProportion}.png')
        plt.show()   


    for i, movieConsiderProportion in enumerate(venueConsiderProportions):
        
        plt.plot(subsetSizes, authorsCoveredCountsCentral, label='Centralized Greedy',color = 'black')
        for j,sampleProportion in enumerate(subSampleProportions):
            if movieConsiderProportion == 0.9 and sampleProportion >0.01:
                continue
            distributedMeanRatings = np.mean(authorsCoveredCountsTotalDistributed[sampleProportion], axis=0)
            distributedStdRatings = np.std(authorsCoveredCountsTotalDistributed[sampleProportion], axis=0)
            plt.plot(subsetSizes, distributedMeanRatings, label=f'Distributed Full Bit Sample K = {sampleProportion*100}%',color = colors[2*j],marker='o')
            plt.fill_between(subsetSizes, distributedMeanRatings - distributedStdRatings, distributedMeanRatings + distributedStdRatings, alpha=0.1,color = colors[2*j])

            lowBitRatings = authorsCoveredCountsLowBitTotal[(sampleProportion, movieConsiderProportion)]
            meanLowBitRatings = np.mean(lowBitRatings, axis=0)
            stdLowBitRatings = np.std(lowBitRatings, axis=0)
            
            plt.plot(subsetSizes, meanLowBitRatings, label=f'Distributed Low Bit Sample K = {sampleProportion*100}%',color = colors[2*j+1],marker='o')
            plt.fill_between(subsetSizes, meanLowBitRatings - stdLowBitRatings, meanLowBitRatings + stdLowBitRatings, alpha=0.1,color = colors[2*j+1])
        
        plt.title(f'Movie Sampling d = {movieConsiderProportion*100}%')
        plt.xlabel('Number of Movies')
        plt.ylabel('Function Value')
        plt.legend()
        plt.savefig(f'algorithm_3_graphs/DBLP_varying_k_at_fixed_d={movieConsiderProportion}.png')
        plt.show()



    


