import scipy as sp
import numpy as np
import scipy.linalg as linalg

'''############################'''
'''Moore Penrose Pseudo Inverse'''
'''############################'''

'''
Compute Moore Penrose Pseudo Inverse
Input: X: matrix to invert
       tol: tolerance cut-off to exclude tiny singular values (default=1e15)
Output: Pseudo-inverse of X.
Note: Do not use scipy or numpy pinv method. Implement the function yourself.
      You can of course add an assert to compare the output of scipy.pinv to your implementation
'''
def compute_pinv(X=None,tol=1e-15):
    L,D,R=linalg.svd(X)
    S=np.zeros([X.shape[0],X.shape[1]])
    D_plus=D[D>tol]**-1
    for i in range(X.shape[1]):
        S[i][i]=D_plus[i]
    inverse=np.dot(L,np.dot(S,R)).T
    return inverse
    pass
