"""
Course  : Data Mining II (636-0019-00L)
"""
from utils import *
from pca import *
from impute import *

import scipy.misc as misc
from sklearn.datasets import make_moons
from sklearn.decomposition import KernelPCA
'''
Main Function
'''
if __name__ in "__main__":
    #Initialise plotting defaults
    initPlotLib()

    ##################
    #Exercise 1:
    #Load lena
    try: # Lena has been removed in new versions of scipy.
        #if scipy is new use new Ascent image
        X = misc.lena()
    except:
        X = misc.ascent()
 
    #generate data matrix with 75% missing values
    X_missing = randomMissingValues(X,per=0.75)
    
    #plot data for comparison
    pl.figure(figsize=(8,4))
    pl.subplot(121)
    pl.gray()
    pl.imshow(X)
    pl.title("Original Image")
    pl.subplot(122)
    pl.gray()
    pl.imshow(X_missing)
    pl.title("Image with 75% Missing Data")
    pl.savefig("exercise1_1.pdf")

    #Impute data with optimal rank r
    #TODO Implement svd_imputation_optimised
    [X_imputed,r,testing_errors] = svd_imputation_optimised(X=X_missing,
                                                            ranks=sp.arange(1,30),
                                                            test_size=0.2)
    #plot data for comparison
    pl.figure(figsize=(12,4))
    pl.subplot(131)
    pl.gray()
    pl.imshow(X)
    pl.title("Original Image")
    pl.subplot(132)
    pl.gray()
    pl.imshow(X_missing)
    pl.title("Image with 75% Missing Data")
    pl.subplot(133)
    pl.gray()
    pl.imshow(X_imputed)
    pl.title("SVD Imputed Image")
    pl.savefig("exercise1_2.pdf")

    #Plot testing_error and highlight optimial rank r
    pl.figure(figsize=(4,5))
    #TODO Plot testing_errors and highlight optimal rank r
    pl.plot(testing_errors)
    pl.title('Testing error')
    pl.plot(r,min(testing_errors),marker='o')
    pl.annotate('optimal rank r',xy=(r,min(testing_errors)),arrowprops=dict(arrowstyle='->'), xytext=(14, 1200))
    pl.savefig("exercise1_3.pdf")
    
    #Exercise 2
    #load data
    [X,y] = make_moons(n_samples=500,noise=None)
    #Perform a PCA
    #1. Compute covariance matrix
    cov=computeCov(X)
    #2. Compute PCA by computing eigen values and eigen vectors
    eigen_values,eigen_vectors=computePCA(cov)
    #3. Transform your input data onto a 2-dimensional subspace using the first two PCs
    transformed=transformData(eigen_vectors[:,:2],X)
    #4. Plot your transformed data and highlight the three different sample classes
    plotTransformedData(transformed,labels=y,filename='exercise2_1.pdf')
    #5. How much variance can be explained with each principle component?
    sp.set_printoptions(precision=2)
    var = computeVarianceExplained(eigen_values)
    print("Variance Explained Exercise 2a: ")
    for i in range(var.shape[0]):
        print("PC %d: %.2f"%(i+1,var[i]))

    #1. Perform Kernel PCA
    transformed1=RBFKernelPCA(X,gamma=1,n_components=2)
    transformed5=RBFKernelPCA(X,gamma=5,n_components=2)
    transformed10=RBFKernelPCA(X,gamma=10,n_components=2)
    transformed20=RBFKernelPCA(X,gamma=20,n_components=2)
    #2. Plot your transformed data and highlight the three different sample classes
    plotTransformedData(transformed1,labels=y,filename='exercise2_3_gamma1.pdf')
    plotTransformedData(transformed5,labels=y,filename='exercise2_3_gamma5.pdf')
    plotTransformedData(transformed10,labels=y,filename='exercise2_3_gamma10.pdf')    
    plotTransformedData(transformed20,labels=y,filename='exercise2_3_gamma20.pdf')

    


