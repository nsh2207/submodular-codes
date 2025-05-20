from indexed_pq import IndexedMaxPQ
import sys
import matplotlib.pyplot as plt
import random
from centralized_greedy import compute_greedy_maximal_set_central
import numpy as np


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

def compute_greedy_maximal_set(graph, transposeGraph,  subsetSize, subSet, subSampleProportion):
    '''
    This is the main function. It takes the graph, the transpose graph, the subset size, the subset of venues already included, and the sub sample proportion as input 
    and returns the subset of venues that maximizes the number of authors covered by adding the new venues to subSet parameter passed and the count of total authors covered by the set.
    
    It actually doesn't use the transposeGraph parameter in the computation of next optimal member, but it is used at the end
    to get the count of total authors covered by the subset of venues collected till now.
    '''
    for t in range(subsetSize):
        subGraph = subSampleGraphGenerator(graph,subSampleProportion) # Generate the sub sampled graph
        
        
        # If the author has already been covered by the venues in the subSet, then remove the author from the subGraph

        authors_to_remove = []

        for author, venues in subGraph.items():
            if set(venues).intersection(subSet): 
                authors_to_remove.append(author)
        
        for author in authors_to_remove:
            subGraph.pop(author)

        
        # Generate the transpose graph of the sub sampled graph  
        transpose_graph = transpose_graph_generator(subGraph)

        
        # Delete the venues already included in the subSet from the transpose graph obtained in the previous step
        for includedVenue in subSet:
            if includedVenue in transpose_graph:
                del transpose_graph[includedVenue]
        
        # Find the venue with the maximum degree in the transpose graph

        maxDegree = 0 # Maximum degree of the venue in the transpose graph such that the venue is not already included in the subSet and the authors covered by the venue are not already covered by the venues in the subSet
        nextVenue = None # The venue with the maximum degree in the transpose graph such that the venue is not already included in the subSet and the authors covered by the venue are not already covered by the venues in the subSet
        for venue in transpose_graph:
            if len(transpose_graph[venue]) > maxDegree:
                maxDegree = len(transpose_graph[venue])
                nextVenue = venue
        subSet.append(nextVenue) # Add the venue with the maximum degree to the subSet
    
    
    coveredAuthorsSet = set() # Set of authors covered by the venues in the subSet obtained till now
    for venue in subSet:
        coveredAuthorsSet = coveredAuthorsSet.union(transposeGraph[venue])
    
    return subSet, len(coveredAuthorsSet) # Return the subSet and the count of total authors covered by the venues in the subSet till now
            



def randomSubsetGenerator(transposeGraph,subsetSize, subSet):
    '''
    Takes the graph and the subset size as input and returns a random subset of venues of the given size.
    '''
    setOfVenues = set(transposeGraph.keys())
    for t in range(subsetSize):
        nextVenue = random.choice(list(setOfVenues.difference(subSet)))
        subSet.append(nextVenue)
    authorsCovered = set()
    for venue in subSet:
        authorsCovered = authorsCovered.union(transposeGraph[venue])
    return subSet,len(authorsCovered)


