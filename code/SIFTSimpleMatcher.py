# -*- coding: utf-8 -*-
"""
Created on Sat Aug 26 21:36:46 2017

@author: HGY
"""

import numpy as np
from scipy.io import loadmat


#%% SIFTSimpleMatcher function
def SIFTSimpleMatcher(descriptor1, descriptor2, THRESH=0.70):
    '''
    SIFTSimpleMatcher 
    Match one set of SIFT descriptors (descriptor1) to another set of
    descriptors (decriptor2). Each descriptor from descriptor1 can at
    most be matched to one member of descriptor2, but descriptors from
    descriptor2 can be matched more than once.
    
    Matches are determined as follows:
    For each descriptor vector in descriptor1, find the Euclidean distance
    between it and each descriptor vector in descriptor2. If the smallest
    distance is less than thresh*(the next smallest distance), we say that
    the two vectors are a match, and we add the row [d1 index, d2 index] to
    the "match" array.
    
    INPUT:
    - descriptor1: N1 * 128 matrix, each row is a SIFT descriptor.
    - descriptor2: N2 * 128 matrix, each row is a SIFT descriptor.
    - thresh: a given threshold of ratio. Typically 0.7
    
    OUTPUT:
    - Match: N * 2 matrix, each row is a match. For example, Match[k, :] = [i, j] means i-th descriptor in
        descriptor1 is matched to j-th descriptor in descriptor2.
    '''

    #############################################################################
    #                                                                           #
    #                              YOUR CODE HERE                               #
    #                                                                           #
    #############################################################################
    match = []
    N1, N2 = descriptor1.shape[0], descriptor2.shape[0]
    
    for i in range(N1):
        diff = np.tile(descriptor1[i, :] , (N2,1)) - descriptor2
        distance = np.sqrt(np.sum(np.square(diff), axis=1))
        
        distance_order = np.sort(distance)
        if distance_order[0] < distance_order[1] * THRESH:
            match.append([i, np.argmin(distance)])

    #############################################################################
    #                                                                           #
    #                             END OF YOUR CODE                              #
    #                                                                           #
    #############################################################################   
    
    return match
