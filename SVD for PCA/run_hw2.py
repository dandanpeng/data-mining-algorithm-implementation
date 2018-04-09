#import all necessary functions
from utils import *
from pca import *
from pinv import *

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
    #Perform a PCA using covariance matrix and eigen-value decomposition
    #1. Compute covariance matrix
    cov=computeCov(data.data)
    #2. Compute PCA by computing eigen values and eigen vectors
    eigen_values,eigen_vectors=computePCA(cov)
    #3. Transform your input data onto a 2-dimensional subspace using the first two PCs
    transform=transformData(eigen_vectors[:,0:2],data.data)
    #4. Plot your transformed data and highlight the three different sample classes
    plotTransformedData(transform,data.target,filename="exercise1.pdf")
    #5. How much variance can be explained with each principle component?
    var = computeVarianceExplained(eigen_values)
    print("Variance Explained PCA: ")
    for i in range(var.shape[0]):
        print("PC %d: %.2f"%(i+1,var[i]))
    print
    print("Eigen Vectors PCA:")
    print(eigen_vectors)
    print

    #Perform a PCA using SVD
    #1. Normalise data by substracting the mean
    norm=zeroMean(data.data)
    #2. Compute PCA by computing eigen values and eigen vectors
    eigen_values,eigen_vectors=computePCA_SVD(data.data)
    #3. Transform your input data onto a 2-dimensional subspace using the first two PCs
    transform=transformData(eigen_vectors[:,0:2],data.data)
    #4. Plot your transformed data and highlight the three different sample classes
    plotTransformedData(transform,data.target,filename="exercise2d.pdf")
    #5. How much variance can be explained with each principle component?
    var = computeVarianceExplained(eigen_values)
    print("Variance Explained SVD: ")
    for i in range(var.shape[0]):
        print("PC %d: %.2f"%(i+1,var[i]))
    print
    print("Eigen Vectors SVD:")
    print(eigen_vectors)
    print


    #Exercise 3
    #1. Compute the Moore-Penrose Pseudo-Inverse on the iris data
    inverse=compute_pinv(data.data)    
    #2. Check Properties
    status = (np.round(np.dot(data.data,np.dot(inverse,data.data)),2)==data.data).all()
    print("X X^+ X = X is ", status)
    status = (np.round(np.dot(inverse,np.dot(data.data,inverse)),7)==np.round(inverse,7)).all()
    print("X^+ X X^+ = X^+ is ", status)
