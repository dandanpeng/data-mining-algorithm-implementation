"""
Course  : Data Mining II (636-0019-00L)
"""
import scipy as sp
import scipy.linalg as linalg
import pylab as pl

from utils import plot_color, plot_markers

'''
Compute Distance Matrix using Euclideans Distance
Input: matrix of size n x m, where n is the number of samples and m the number of features
Output: distance matrix of size n x n 
'''

def computeEuclideanDistanceMatrix(matrix=None):
    dist_matrix = sp.zeros([matrix.shape[0],matrix.shape[0]])
    for i in range(len(matrix)):
        for j in range(len(matrix)):
            dist_matrix[i,j]=sp.sqrt(((matrix[i,:]-matrix[j,:]).T).dot(matrix[i,:]-matrix[j,:]))
    return dist_matrix
    pass

'''
Classical Metric Multidimensional Scaling
Input: matrix: distance matrix of size n x n
       n_components: number of dimensions/components to return
Output: transformed_data of size n x n_components
'''
def classicalMDS(matrix=None,n_components=2):
    n = matrix.shape[0]
    H = sp.eye(n)-(sp.ones([n,n])).dot(sp.ones([n,n]).T)/n
    A = -0.5*(matrix**2)
    B = (H.dot(A)).dot(H)
    eigenvalues,eigenvectors = linalg.eig(B)
    indices = sp.argsort(-eigenvalues)
    eigenvalues,eigenvectors = [sp.real(eigenvalues[indices]), eigenvectors[:,indices]]
    return eigenvectors[:,:n_components].dot(sp.diag(sp.sqrt(eigenvalues[:n_components])))
    pass

'''
Plot cities as dots and add name tag to city
Input: transformed_data: transformed data of size n x 2
       names: array of city_names
       filename: filename to save image
'''
def plotCities(transformed_data=None,names=None,filename="cities.pdf"):
    fig,ax = pl.subplots()
    ax.scatter(transformed_data[:,0],transformed_data[:,1],color = plot_color)
    for i,name in enumerate(names):
        ax.annotate(name,(transformed_data[i,0],transformed_data[i,1]))
    pl.savefig(filename)
    pass


