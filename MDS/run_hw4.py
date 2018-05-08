"""
Course  : Data Mining II (636-0019-00L)
"""
from utils import *
from pca import *
from mds import *

'''
Main Function
'''
if __name__ in "__main__":
    #Initialise plotting defaults
    initPlotLib()

    #MDS Example on City Distances
    print("Exercise 1 ...")
    #transform data with MDS and plot transformed City data
    transformed_data = classicalMDS(city_dist)
    plotCities(transformed_data,names=city_names,filename="cities.pdf")
    #MDS on real world data
    #load indel and country data
    print("Loading Data for Exercise 2 ...")
    indels = sp.loadtxt("indels.csv",delimiter=",")
    countries = sp.loadtxt("countries.csv",dtype="|S14", delimiter="\t")
    #1. Perform MDS 
    #1.1 Compute Euclidean Distance matrix on indels matrix
    print("Computing eucleidean distance matrix for exercise 2.1 ...")
    dist_matrix = computeEuclideanDistanceMatrix(indels)
    #1.2 Compute MDS
    print("Computing MDS and plotting transformed data ...")
    transformed_data = classicalMDS(dist_matrix)
    #1.3 Plot transformed MDS results
    plotTransformedData(transformed_data,labels=countries,filename='countries.pdf')
    
    #2. Perform classical PCA
    print("Performing FastPCA and plotting transformed data ...")
    transformed_data = computeFastPCA(indels)
    plotTransformedData(transformed_data,labels=countries,filename='FastPCA.pdf')

