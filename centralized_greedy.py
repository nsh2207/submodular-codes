from indexed_pq import IndexedMaxPQ
import sys
import matplotlib.pyplot as plt

def read_graph(file_path):
    '''
    This function reads the graph from the file and returns the graph as a dictionary with keys as authors and values as the set of venues
    as well as the transpose of the graph i.e. the graph with the edges reversed i.e. venues as keys and the values as the set of authors covered by the corresponding venue
    '''
    graph = {} # Dictionary to store the graph with authors as keys and the set of venues as values
    transpose_graph = {} # Transpose of the graph i.e. the graph with the edges reversed 
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split(':')
            node = parts[0].strip()
            neighbors = [author.strip() for author in parts[1][1:-1].split(',')]
            graph[node] = set(neighbors)
            for neighbor in neighbors:
                if neighbor not in transpose_graph:
                    transpose_graph[neighbor] = set()
                transpose_graph[neighbor].add(node)
    return graph,transpose_graph

def compute_greedy_maximal_set_central(graph,transpose_graph, subsetSize):
    '''
    This function takes the graph, the transpose graph, and the subset size as input 
    and returns the subset of venues that maximizes the number of authors covered by adding the new venues to subSet parameter passed.
    '''
    venueDegrees = {} # Dictionary to store the degree of each venue i.e. the number of authors covered by the venue
    for venue,authors in transpose_graph.items():
        # this is done from transpose_graph because we are interested in the number of authors covered by the venue 
        venueDegrees[venue] = len(authors) 
    
    # Maximum Indexed Priority Queue data structure to store the venues with their degrees and update the degrees of the venues in O(logN) time
    # as the covered authors are removed from the graph in each iteration i.e. new inclusion of a venue in the subset
    pq = IndexedMaxPQ(20000)
    for venue,degree in venueDegrees.items():
        pq.insert(int(venue),degree) # Insert the venue with its degree in the indexed priority queue with the key as the venue and the value as the degree
    
    
    topVenues = [] # List to store the top venues that maximize the number of authors covered
    
    for i in range(subsetSize):
        topVenue = str(pq.pq[1]) # Get the venue name with the maximum degree of uncovered authors
        degree= pq.deleteMax()[0] # Get the degree of the venue with the maximum degree of uncovered authors
        topVenues.append((topVenue,degree)) # Appending the venue and the number of newly covered authors it is covering to the topVenues list

        # Block of code to clear up the graph by removing the authors covered by the topVenue and the top Venue and update the degrees of the venues in the indexed priority queue

        authorsToBeRemoved = transpose_graph[topVenue] # Get the authors covered by the topVenue and they will be removed from the graph

        # As we exclude the newly covered authors by the topVenue from the graph, we need to update the degrees of the remaining venues in the transpose graph and values
        
        prepareToBeSubtractedSets = {} # Dictionary such that keys are the venues whose degrees should be updated and the values are the set of authors to be removed from the venue
        # I am using prepareToBeSubtractedSets dictionary such that for each venue in which the newly covered authors are participating, I can 
        # " prepare the set of authors to be subtracted " from the venue and then update the degrees of those venues in the indexed priority queue
        
        # Block of code to populate the prepareToBeSubtractedSets dictionary and remove the authors from the graph

        for author in authorsToBeRemoved:
            for venue in graph[author]:
                if venue not in transpose_graph:
                    continue
                if venue not in prepareToBeSubtractedSets:
                    prepareToBeSubtractedSets[venue] = set()
                prepareToBeSubtractedSets[venue].add(author)
            graph.pop(author)
        
        # Block of code to remove the set of covered authors from transpose graph and update the degrees of the venues in the indexed priority queue 
        for subtractedVenue,authors in prepareToBeSubtractedSets.items():
            transpose_graph[subtractedVenue] = transpose_graph[subtractedVenue]-authors
            pq.decreaseKey(int(subtractedVenue),len(transpose_graph[subtractedVenue]))
        transpose_graph.pop(str(topVenue))  
    return topVenues # Return the top venues that maximize the number of authors covered along with the exclusive degrees of theses venues
    


if __name__ == "__main__":
    file_path = 'graph.txt'
    print("Reading the graph...")
    subsetSizes = [size for size in range(5, 31, 5)]
    authorsCoveredCounts = [] # List to store the percentage of authors covered by the top venues for each subset size
    graph,transpose_graph = read_graph(file_path) # Generate the graph and the transpose graph
    authorsCovered = 0     # Number of authors covered by the top venues till now
    totalAuthors = len(graph)    # Total number of authors in the graph
    for subsetSize in subsetSizes:   

        # Though subsetSize is 5,10,15,20,... I am passing 5 as the subsetSize parameter to the function because 
        # whole logic is incremental and the function will add 5 venues to the subset each time it is called and the 
        # size of set collected till now from beginning is subsetSize

        venuesList = compute_greedy_maximal_set_central(graph,transpose_graph, subsetSize=5) # Get the top venues
        
        for venue in venuesList:
            authorsCovered += venue[1]
        authorsCoveredCounts.append(100*authorsCovered/totalAuthors)
    
    # Block to plot the graph of the percentage of authors covered by the top venues for each subset size for only centralized greedy algorithm

    plt.xlabel('Subset Size')
    plt.ylabel('Percentage of Authors Covered')
    plt.title('Authors Covered by Top Venues')
    plt.plot(subsetSizes, authorsCoveredCounts, marker='+', color='r', linestyle='None')
    plt.plot(subsetSizes, authorsCoveredCounts, color='b')
    plt.savefig('centralized_mean_dblp.png')
    plt.show()
    



    