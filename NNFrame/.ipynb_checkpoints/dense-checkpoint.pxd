# cython: language_level=3

import numpy as np
cimport numpy as np
from layer cimport Layer

cdef class Dense(Layer):
    # Using C arrays for fast access
    cdef double* weights_data
    cdef double* bias_data
    cdef double* weights_grad_data
    cdef double* input_grad_data

    # Memory Views for convenient access
    cdef public double[:, :] weights
    cdef public double[:, :] bias
    cdef public double[:, :] weights_gradient
    cdef public double[:, :] input_gradient

    # Numpy arrays for memory management
    cdef np.ndarray weights_array
    cdef np.ndarray bias_array
    cdef np.ndarray weights_grad_array
    cdef np.ndarray input_grad_array
    
    # Diemnsion
    cdef public int input_size
    cdef public int output_size