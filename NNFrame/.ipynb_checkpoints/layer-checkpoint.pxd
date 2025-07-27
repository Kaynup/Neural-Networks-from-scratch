# cython: language_level=3

import numpy as np
cimport numpy as np

cdef class Layer:
    cdef public double[:, :] input
    cdef public double[:, :] output
    cdef public int input_size
    cdef public int output_size

    cpdef double[:, :] forward(self, double[:, :] input)
    cpdef double[:, :] backward(self, double[:, :] output_gradient, double learning_rate)