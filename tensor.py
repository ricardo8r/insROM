import os,sys
import re
import numpy as np

def unfolding_order(A, n):
    return [i for i in range(n+1, A.ndim)] + [i for i in range(n)]

def unfolding_matrix_size(A, n):
    row_count = A.shape[n]    
    col_count = 1   
    for i in xrange(A.ndim):
        if i != n: col_count *= A.shape[i]        
    return (row_count, col_count)

def unfolding_stride(A, mode_order):
    stride = [0 for i in xrange(A.ndim)]
    stride[mode_order[A.ndim-2]] = 1
    for i in xrange(A.ndim-3, -1, -1):
        stride[mode_order[i]] = (
            A.shape[mode_order[i+1]] * stride[mode_order[i+1]])
    return stride

def tensor_indices(r, c, A, n, mode_order, stride):
    i = [0 for j in xrange(A.ndim)]
    i[n] = r
    i[mode_order[0]] = c / stride[mode_order[0]]
    for k in xrange(1, A.ndim-1):
        i[mode_order[k]] = (
            (c % stride[mode_order[k-1]]) / stride[mode_order[k]])
    return i

A = np.zeros((3,2,5,4))
n=1
mode_order = unfolding_order(A, n)
print(mode_order)

