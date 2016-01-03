"""This is a tested unit of a program; 
the first function transforms the data, and
the second function computes top k components of the data.
"""

import numpy as np
from numpy.linalg import eigh

def estimateCovariance(data):
    """Compute the covariance matrix for a given rdd.

    Note:
        The multi-dimensional covariance array should be calculated using outer products.  Don't
        forget to normalize the data by first subtracting the mean.

    Args:
        data (RDD of np.ndarray):  An `RDD` consisting of NumPy arrays.

    Returns:
        np.ndarray: A multi-dimensional array where the number of rows and columns both equal the
            length of the arrays in the input `RDD`.
    """
    mean = data.sum()/data.count()
    dataZeroMean = data.map(lambda a: a-mean)
    return dataZeroMean.map(lambda a: np.outer(a.T,a)).sum()/data.count()

def pca(data, k=2):
    """Computes the top `k` principal components, corresponding scores, and all eigenvalues.

    Note:
        All eigenvalues should be returned in sorted order (largest to smallest). `eigh` returns
        each eigenvectors as a column.  This function should also return eigenvectors as columns.

    Args:
        data (RDD of np.ndarray): An `RDD` consisting of NumPy arrays.
        k (int): The number of principal components to return.

    Returns:
        tuple of (np.ndarray, RDD of np.ndarray, np.ndarray): A tuple of (eigenvectors, `RDD` of
            scores, eigenvalues).  Eigenvectors is a multi-dimensional array where the number of
            rows equals the length of the arrays in the input `RDD` and the number of columns equals
            `k`.  The `RDD` of scores has the same number of rows as `data` and consists of arrays
            of length `k`.  Eigenvalues is an array of length d (the number of features).
    """
    coVar=estimateCovariance(data)
    eigVals, eigVecs = eigh(coVar)
    inds = np.argsort(eigVals)
    
    
    
    topVecs=[]
    for i in range(k):
      topVecs.append(np.amax(inds))
      inds[np.amax(inds)]=-1
    
    #print 'ind:', inds
    topComps=[]
    for i in topVecs:
      topComps.append(eigVecs[:,i])
      
    kComp=np.array(topComps)
    
    #print 'compon:', kComp
    #print 'first:', data.take(1)
    scores=data.map(lambda a: np.dot(a, kComp.T))
    # Return the `k` principal components, `k` scores, and all eigenvalues
    eigVals=eigVals[::-1]
    return kComp.T, scores, eigVals
