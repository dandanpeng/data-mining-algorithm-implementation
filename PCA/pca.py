"""
Homework: Principal Component Analysis
Course  : Data Mining II (636-0019-00L)
"""
import scipy as sp
import scipy.linalg as linalg
import pylab as pl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils import plot_color

'''############################'''
'''Principle Component Analyses'''
'''############################'''

'''
Compute Covariance Matrix
Input: Matrix of size #samples x #features
Output: Covariance Matrix of size #features x #features
Note: Do not use scipy or numpy cov. Implement the function yourself.
      You can of course add an assert to check your covariance function
      with those implemented in scipy/numpy.
'''
def computeCov(X=None):
    # Please fill this function
    samples=X.shape[0]
    features=X.shape[1]
    cov=np.zeros(shape=(features,features))
    for i in range(features):
        for j in range(features):
            	cov[i,j]=np.dot(X[:,i]-np.mean(X[:,i]),X[:,j]-np.mean(X[:,j]))/(samples-1)
    return cov
    pass

'''
Compute PCA
Input: Covariance Matrix
Output: [eigen_values,eigen_vectors] sorted in such a way that eigen_vectors[:,0] is the first principle component
        eigen_vectors[:,1] the second principle component etc...
Note: Do not use an already implemented PCA algorithm. However, you are allowed to use an implemented solver 
      to solve the eigenvalue problem!
'''
def computePCA(matrix=None):
	# Please fill this function
    [eigen_values,eigen_vectors]=linalg.eig(matrix)
    sorting_indices=sp.argsort(eigen_values)[::-1]
    sorted_eigen_values=eigen_values[sorting_indices]
    sorted_eigen_vectors=eigen_vectors[:,sorting_indices]
    return sorted_eigen_values,sorted_eigen_vectors
    pass

'''
Transform Input Data Onto New Subspace
Input: pcs: matrix containing the first x principle components
       data: input data which should be transformed onto the new subspace
Output: transformed input data. Should now have the dimensions #samples x #components_used_for_transformation
'''
def transformData(pcs=None,data=None):
    # Please fill this function
    pcs=pcs[:,0:2]
    data=data-data.mean(axis=0)
    return np.dot(data,pcs)
    pass

'''
Compute Variance Explaiend
Input: eigen_values
Output: return vector with varianced explained values. Hint: values should be between 0 and 1 and should sum up to 1.
'''
def computeVarianceExplained(evals=None):
    # Please fill this function
    return (evals/sum(evals))
    pass

'''############################'''
'''Different Plotting Functions'''
'''############################'''

'''
Plot Cumulative Explained Variance
Input: var: variance explained vector
       filename: filename to store the file
'''
def plotCumSumVariance(var=None,filename="cumsum.pdf"):
    #PLOT FIGURE
    plt.figure()
    #You can use plot_color[] to obtain different colors for your plots
    plt.plot(range(1,var.shape[0]+1),np.cumsum(var))
    plt.xlabel('principle components')
    plt.ylabel('Cumulative Sum Variance')
    #Save file
    pl.savefig(filename)

'''
Plot Transformed Data
Input: transformed: data matrix (#sampels x 2)
       labels: target labels, class labels for the samples in data matrix
       filename: filename to store the plot
'''
def plotTransformedData(transformed=None,labels=None,filename="exercise1.pdf"):
    #PLOT FIGURE
    plt.figure()
    #You can use plot_color[] to obtain different colors for your plots
    df=pd.DataFrame(transformed)
    df['labels']=labels
    group=df.groupby('labels')
    for i in group.groups.keys():
        plt.scatter(df.iloc[group.groups[i],0],df.iloc[group.groups[i],1],color=plot_color[int(i)],label='type'+str(i))
    plt.legend()
    #plt.scatter(transformed[:,0], transformed[:,1],c=labels)
    #Save File
    pl.savefig(filename)

'''############################'''
'''Data Preprocessing Functions'''
'''############################'''

'''
Exercise 2
Data Normalisation (Zero Mean, Unit Variance)
'''
def dataNormalisation(X=None):
    # Please fill this function
    return (X-X.mean(axis=0))/X.std(axis=0)
    pass
