"""
Homework: Principal Component Analysis
Course  : Data Mining II (636-0019-00L)
"""

#import all necessary functions
from utils import *
from pca import *

'''
Main Function
'''
if __name__ in "__main__":
    #Initialise plotting defaults
    initPlotLib()

    ##################
    #Exercise 1:

    #Load Iris data
    data = loadIrisData()
    
    #Perform a PCA
    #1. Compute covariance matrix
    cov=computeCov(data.data)
    #2. Compute PCA by computing eigen values and eigen vectors
    eigen_values,eigen_vectors=computePCA(cov)
    #3. Transform your input data onto a 2-dimensional subspace using the first two PCs
    trans_data=transformData(eigen_vectors,data.data)
    #4. Plot your transformed data and highlight the three different sample classes
    plotTransformedData(trans_data,data.target,filename="exercise1.pdf")
    #5. How much variance can be explained with each principle component?
    var = sp.array(computeVarianceExplained(eigen_values))#Compute Variance Explained
    print("Variance Explained Exercise 1: ")
    for i in range(var.shape[0]):
        print("PC %d: %.2f"%(i+1,var[i]))
    print

    ##################
    #Exercise 2:
    
    #Simulate Data
    data = simulateData()
    #Perform a PCA
    #1. Compute covariance matrix
    cov2=computeCov(data.data)
    #2. Compute PCA by computing eigen values and eigen vectors
    eigen_values,eigen_vectors=computePCA(cov2)
    #3. Transform your input data onto a 2-dimensional subspace using the first two PCs
    trans_data=transformData(eigen_vectors,data.data)
    #4. Plot your transformed data and highlight the three different sample classes
    plotTransformedData(trans_data,data.target,filename="exercise2.pdf")
    #5. How much variance can be explained with each principle component?
    var = sp.array(computeVarianceExplained(eigen_values)) #Compute Variance Explained
    print("Variance Explained Exercise 2.1: ")
    for i in range(15):
        print("PC %d: %.2f"%(i+1,var[i]))
    #6. Plot cumulative variance explained per PC
    plotCumSumVariance(var,filename="cumsum.pdf")
    
    ##################
    #Exercise 2 Part 2:
    
    #1. normalise data
    norm_data=dataNormalisation(data.data)
    #2. compute covariance matrix
    cov2_norm=computeCov(norm_data)
    #3. compute PCA
    norm_eigen_values,norm_eigen_vectors=computePCA(cov2_norm)
    #4. Transform your input data inot a 2-dumensional subspace using the first two PCs
    norm_trans_data=transformData(norm_eigen_vectors,norm_data)
    #5. Plot your transformed data
    plotTransformedData(norm_trans_data,data.target,filename="normalized_exercise2.pdf")
    #6. Compute Variance Explained
    norm_var = sp.array(computeVarianceExplained(norm_eigen_values)) #Compute Variance Explained
    print("Variance Explained Exercise 2.2: ")
    for i in range(15):
        print("PC %d: %.2f"%(i+1,norm_var[i]))
    #7. Plot Cumulative Variance
    plotCumSumVariance(norm_var,filename="norm_cumsum.pdf")
