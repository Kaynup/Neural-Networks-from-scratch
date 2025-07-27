# cython: language_level=3

import numpy as np
cimport numpy as np

cdef class Layer:
    def __init__(self):
        self.input_size = 0
        self.output_size = 0

    cpdef double[:, :] forward(self, double[:, :] input):
        """Base forward pass - stores input and returns it unchanged"""
        self.input = input
        self.output = input
        return self.output

    cpdef double[:, :] backward(self, double[:, :] output_gradient, double learning_rate):
        """Base backward pass - return gradient unchanged"""
        return output_gradient