"""
Course  : Data Mining II (636-0019-00L)
"""
import scipy as sp
import scipy.linalg as linalg
import scipy.spatial.distance as distance
import pylab as pl

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
    Xm = X - X.mean(axis=0)
    return 1.0/(Xm.shape[0]-1)*sp.dot(Xm.T,Xm)

'''
Compute PCA
Input: Covariance Matrix
Output: [eigen_values,eigen_vectors] sorted in such a why that eigen_vectors[:,0] is the first principle component
        eigen_vectors[:,1] the second principle component etc...
Note: Do not use an already implemented PCA algorithm. However, you are allowed to use an implemented solver 
      to solve the eigenvalue problem!
'''
def computePCA(matrix=None):
    #compute eigen values and vectors
    [eigen_values,eigen_vectors] = linalg.eig(matrix)
    #sort eigen vectors in decreasing order based on eigen values
    indices = sp.argsort(-eigen_values)
    return [sp.real(eigen_values[indices]), eigen_vectors[:,indices]]

'''
Compute PCA using SVD
Input: Data Matrix
Output: [eigen_values,eigen_vectors] sorted in such a why that eigen_vectors[:,0] is the first principle component
        eigen_vectors[:,1] the second principle component etc...
Note: Do not use an already implemented PCA algorithm. However, you are allowed to use SciPy svd solver!
'''
def computePCA_SVD(matrix=None):
    X = 1.0/sp.sqrt(matrix.shape[0]-1) * matrix
    [L,S,R] = linalg.svd(X)
    eigen_values = S*S
    eigen_vectors = R.T
    return [eigen_values,eigen_vectors]

'''
Compute Fast PCA
Input: Data Matrix of size n x m
       n_components: number of compontents to use for transformation of data
Output: [eigen_values,transformed_data] 
'''
def computeFastPCA(matrix=None,n_components=2):
    matrix= matrix - matrix.mean(axis=0)
    K = (matrix).dot(matrix.T)
    eigen_values,eigen_vectors = linalg.eig(K)
    indices = sp.argsort(-eigen_values)
    eigen_values,eigen_vectors = [sp.real(eigen_values[indices]), eigen_vectors[:,indices]]
    v=sp.zeros([len(matrix.T.dot(eigen_vectors[:,0])),n_components])
    for i in range(n_components):
        v[:,i] = (matrix.T).dot(eigen_vectors[:,i])/linalg.norm((matrix.T).dot(eigen_vectors[:,i]))
    return transformData(v,matrix)
    pass 

'''
Compute Kernel PCA
Input: 
Output: [eigen_values,eigen_vectors] sorted in such a why that eigen_vectors[:,0] is the first principle component, etc...
Note: Do not use an already implemented Kernel PCA algorithm.
'''
def RBFKernelPCA(matrix=None,gamma=1,n_components=2):
    #1. Compute RBF Kernel
    K = sp.exp(float(-gamma) * distance.squareform(distance.pdist(matrix,'sqeuclidean')))
    #2. Center kernel matrix
    n = matrix.shape[0]
    K = (sp.identity(n) - 1.0/n * sp.ones((n,n))).dot(K).dot(sp.identity(n)-1.0/n*sp.ones((n,n)))
    #3. Compute eigenvalues and eigenvactors
    [eigen_values,eigen_vectors] = linalg.eig(K)
    #4. sort eigen vectors in decreasing order based on eigen values
    indices = sp.argsort(-eigen_values)
    eigen_values = eigen_values[indices]
    eigen_vectors = eigen_vectors[:,indices]
    for i,eigen_value in enumerate(eigen_values):
        eigen_vectors[:,i] = sp.sqrt(1.0/eigen_value) * eigen_vectors[:,i]
    eigen_value = eigen_values[:n_components]
    eigen_vectors = eigen_vectors[:,:n_components]
    transformed = sp.zeros((n,n_components))
    for i in xrange(n):
        transformed[i,:] = sp.dot(eigen_vectors.T,K[:,i])
    return transformed
'''
Transform Input Data Onto New Subspace
Input: pcs: matrix containing the first x principle components
       data: input data which should be transformed onto the new subspace
Output: transformed input data. Should now have the dimensions #samples x #components_used_for_transformation
'''
def transformData(pcs=None,data=None):
    return sp.dot(pcs.T,data.T).T

'''
Compute Variance Explaiend
Input: eigen_values
Output: return vector with varianced explained values. Hint: values should be between 0 and 1 and should sum up to 1.
'''
def computeVarianceExplained(evals=None):
    return evals/evals.sum()


'''############################'''
'''Different Plotting Functions'''
'''############################'''

'''
Plot Cumulative Explained Variance
Input: var: variance explained vector
       filename: filename to store the file
'''
def plotCumSumVariance(var=None,filename="cumsum.pdf"):
    pl.figure()
    pl.plot(sp.arange(var.shape[0]),sp.cumsum(var)*100)
    pl.xlabel("Principle Component")
    pl.ylabel("Cumulative Variance Explained in %")
    pl.grid(True)
    #Save file
    pl.savefig(filename)

'''
Plot Transformed Data
Input: transformed: data matrix (#sampels x 2)
       labels: target labels, class labels for the samples in data matrix
       filename: filename to store the plot
'''
def plotTransformedData(transformed=None,labels=None,filename=None):
    pl.figure()
    ind_l = sp.unique(labels)
    legend = []
    for i,label in enumerate(ind_l):
        ind = sp.where(label==labels)[0]
        plot = pl.scatter(transformed[ind,0],transformed[ind,1],color=plot_color[i],alpha=0.5)
        legend.append(plot)
    pl.legend(ind_l,scatterpoints=1,numpoints=1,prop={'size':8},ncol=2,loc="upper right",fancybox=True)
    pl.xlabel("Transformed X Values")
    pl.ylabel("Transformed Y Values")
    pl.grid(True)
    #Save File
    if filename!=None:
       pl.savefig(filename)

'''############################'''
'''Data Preprocessing Functions'''
'''############################'''

'''
Data Normalisation (Zero Mean, Unit Variance)
'''
def dataNormalisation(X=None):
    Xm = X - X.mean(axis=0)
    return Xm/sp.std(Xm,axis=0)

'''
Substract Mean from Data (zero mean)
'''
def zeroMean(X=None):
    return X - X.mean(axis=0)