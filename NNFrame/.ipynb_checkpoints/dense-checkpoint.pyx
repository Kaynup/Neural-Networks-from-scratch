# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

import numpy as np
cimport numpy as np
from layer cimport Layer
from libc.stdlib cimport malloc, free
from libc.string cimport memset
cimport cython

cdef class Dense(Layer):
    def __init__(self, int input_size, int output_size):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        
        # Create numpy arrays for memory management
        self.weights_array = np.random.randn(output_size, input_size).astype(np.float64) * 0.1
        self.bias_array = np.zeros((output_size, 1), dtype=np.float64)
        self.weights_grad_array = np.zeros((output_size, input_size), dtype=np.float64)
        self.input_grad_array = np.zeros((input_size, 1), dtype=np.float64)  # Will be resized as needed
        
        # Create memory views
        self.weights = self.weights_array
        self.bias = self.bias_array
        self.weights_gradient = self.weights_grad_array
        self.input_gradient = self.input_grad_array
        
        # Get pointers to data for fast C-style access
        self.weights_data = <double*>np.PyArray_DATA(self.weights_array)
        self.bias_data = <double*>np.PyArray_DATA(self.bias_array)
        self.weights_grad_data = <double*>np.PyArray_DATA(self.weights_grad_array)
        self.input_grad_data = <double*>np.PyArray_DATA(self.input_grad_array)

    @cython.boundscheck(False)
    @cython.wraparound(False)

    cpdef double[:, :] forward(self, double[:, :] input):
        """Forward pass with optimized C loops"""
        self.input = input

        cdef int batch_size = input.shape[1]
        cdef int i, j, k
        cdef double temp_sum

        # Create output array
        cdef np.ndarray[double, ndim=2] output_array = np.zeros((self.output_size, batch_size), dtype=np.float64)
        cdef double* output_data = <double*>np.PyArray_DATA(output_array)

        # Matrix multiplication: output = weights @ input + bias
        # Using C-style loops for maximum performance
        for i in range(self.output_size):
            for j in range(batch_size):
                temp_sum = 0.0
                for k in range(self.input_size):
                    temp_sum += self.weights_data[i * self.input_size + k] * input[k, j]
                output_data[i * batch_size + j] = temp_sum + self.bias_data[i]
        
        self.output = output_array
        return self.output

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef double[:, :] backward(self, double[:, :] output_gradient, double learning_rate):
        """Backward pass using optimized C loops"""
        cdef int batch_size = output_gradient.shape[1]
        cdef int i, j, k
        cdef double temp_sum
        
        # Resize input gradient array if needed
        if self.input_grad_array.shape[1] != batch_size:
            self.input_grad_array = np.zeros((self.input_size, batch_size), dtype=np.float64)
            self.input_gradient = self.input_grad_array
            self.input_grad_data = <double*>np.PyArray_DATA(self.input_grad_array)
        
        # Compute input gradient: input_grad = weights.T @ output_gradient
        for i in range(self.input_size):
            for j in range(batch_size):
                temp_sum = 0.0
                for k in range(self.output_size):
                    temp_sum += self.weights_data[k * self.input_size + i] * output_gradient[k, j]
                self.input_grad_data[i * batch_size + j] = temp_sum
        
        # Compute weights gradient: weights_grad = output_gradient @ input.T
        memset(self.weights_grad_data, 0, self.output_size * self.input_size * sizeof(double))
        for i in range(self.output_size):
            for j in range(self.input_size):
                temp_sum = 0.0
                for k in range(batch_size):
                    temp_sum += output_gradient[i, k] * self.input[j, k]
                self.weights_grad_data[i * self.input_size + j] = temp_sum / batch_size
        
        # Compute bias gradient: bias_grad = mean(output_gradient, axis=1)
        for i in range(self.output_size):
            temp_sum = 0.0
            for j in range(batch_size):
                temp_sum += output_gradient[i, j]
            self.bias_data[i] -= learning_rate * (temp_sum / batch_size)
        
        # Update weights: weights -= learning_rate * weights_grad
        for i in range(self.output_size):
            for j in range(self.input_size):
                self.weights_data[i * self.input_size + j] -= learning_rate * self.weights_grad_data[i * self.input_size + j]
        
        return self.input_gradient