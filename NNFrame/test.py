import numpy as np
import sys
import traceback

def test_compilation():
    """Test if the modules compile and basic functionality works"""
    print("="*50)
    print("COMPILATION AND BASIC FUNCTIONALITY TEST")
    print("="*50)
    
    try:
        # Test imports
        print("1. Testing imports...")
        from layer import Layer
        from dense import Dense
        print("✓ Imports successful")
        
        # Test basic instantiation
        print("2. Testing instantiation...")
        layer = Dense(10, 5)
        print("✓ Dense layer created successfully")
        
        # Test forward pass
        print("3. Testing forward pass...")
        input_data = np.random.randn(10, 3).astype(np.float64)
        output = layer.forward(input_data)
        print(f"✓ Forward pass successful. Output shape: {output.shape}")
        
        # Test backward pass
        print("4. Testing backward pass...")
        output_grad = np.random.randn(5, 3).astype(np.float64)
        input_grad = layer.backward(output_grad, 0.01)
        print(f"✓ Backward pass successful. Input gradient shape: {input_grad.shape}")
        
        print("\n🎉 ALL TESTS PASSED! Your Cython code is working correctly.")
        return True
        
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        print("\nFull traceback:")
        traceback.print_exc()
        return False


# benchmark_performance.py - Performance comparison
import time
import numpy as np

class NumpyDense:
    """Pure numpy implementation for comparison"""
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(output_size, input_size).astype(np.float64) * 0.1
        self.bias = np.zeros((output_size, 1), dtype=np.float64)
        self.input = None
        
    def forward(self, input_data):
        self.input = input_data
        return self.weights @ input_data + self.bias
    
    def backward(self, output_gradient, learning_rate):
        batch_size = output_gradient.shape[1]
        
        # Gradients
        weights_grad = (output_gradient @ self.input.T) / batch_size
        input_grad = self.weights.T @ output_gradient
        bias_grad = np.mean(output_gradient, axis=1, keepdims=True)
        
        # Updates
        self.weights -= learning_rate * weights_grad
        self.bias -= learning_rate * bias_grad
        
        return input_grad


def benchmark_performance():
    """Compare Cython vs NumPy performance"""
    print("="*50)
    print("PERFORMANCE BENCHMARK")
    print("="*50)
    
    try:
        from dense import Dense as CythonDense
        
        # Test parameters
        input_size = 784
        hidden_size = 512
        batch_sizes = [1, 16, 64, 256]
        num_iterations = 100
        
        print(f"Layer size: {input_size} -> {hidden_size}")
        print(f"Iterations per test: {num_iterations}")
        print("-" * 50)
        
        for batch_size in batch_sizes:
            print(f"\nBatch size: {batch_size}")
            
            # Create layers
            cython_layer = CythonDense(input_size, hidden_size)
            numpy_layer = NumpyDense(input_size, hidden_size)
            
            # Generate test data
            input_data = np.random.randn(input_size, batch_size).astype(np.float64)
            output_grad = np.random.randn(hidden_size, batch_size).astype(np.float64)
            
            # Benchmark Cython implementation
            start_time = time.time()
            for _ in range(num_iterations):
                output = cython_layer.forward(input_data)
                input_grad = cython_layer.backward(output_grad, 0.01)
            cython_time = time.time() - start_time
            
            # Benchmark NumPy implementation
            start_time = time.time()
            for _ in range(num_iterations):
                output = numpy_layer.forward(input_data)
                input_grad = numpy_layer.backward(output_grad, 0.01)
            numpy_time = time.time() - start_time
            
            # Calculate speedup
            speedup = numpy_time / cython_time
            
            print(f"  Cython: {cython_time:.4f}s")
            print(f"  NumPy:  {numpy_time:.4f}s")
            print(f"  Speedup: {speedup:.2f}x {'🚀' if speedup > 1 else '🐌'}")
            
    except ImportError:
        print("❌ Could not import Cython modules. Make sure they compiled successfully.")
    except Exception as e:
        print(f"❌ Benchmark failed: {str(e)}")


# correctness_test.py - Verify mathematical correctness
def test_correctness():
    """Verify that Cython and NumPy implementations give same results"""
    print("="*50)
    print("CORRECTNESS TEST")
    print("="*50)
    
    try:
        from dense import Dense as CythonDense
        
        # Create identical layers
        input_size, output_size, batch_size = 10, 5, 3
        
        # Set random seed for reproducibility
        np.random.seed(42)
        weights_init = np.random.randn(output_size, input_size).astype(np.float64) * 0.1
        bias_init = np.zeros((output_size, 1), dtype=np.float64)
        
        # Create layers with same initial weights
        cython_layer = CythonDense(input_size, output_size)
        numpy_layer = NumpyDense(input_size, output_size)
        
        # Set same weights
        cython_layer.weights[:] = weights_init
        cython_layer.bias[:] = bias_init
        numpy_layer.weights = weights_init.copy()
        numpy_layer.bias = bias_init.copy()
        
        # Test data
        input_data = np.random.randn(input_size, batch_size).astype(np.float64)
        output_grad = np.random.randn(output_size, batch_size).astype(np.float64)
        
        # Forward pass
        cython_output = np.asarray(cython_layer.forward(input_data))
        numpy_output = numpy_layer.forward(input_data)
        
        forward_diff = np.max(np.abs(cython_output - numpy_output))
        print(f"Forward pass max difference: {forward_diff:.2e}")
        
        # Backward pass
        cython_grad = np.asarray(cython_layer.backward(output_grad, 0.01))
        numpy_grad = numpy_layer.backward(output_grad, 0.01)
        
        backward_diff = np.max(np.abs(cython_grad - numpy_grad))
        print(f"Backward pass max difference: {backward_diff:.2e}")
        
        # Check weight updates
        weight_diff = np.max(np.abs(np.asarray(cython_layer.weights) - numpy_layer.weights))
        bias_diff = np.max(np.abs(np.asarray(cython_layer.bias) - numpy_layer.bias))
        
        print(f"Weight update max difference: {weight_diff:.2e}")
        print(f"Bias update max difference: {bias_diff:.2e}")
        
        # Tolerance check
        tolerance = 1e-10
        if all(diff < tolerance for diff in [forward_diff, backward_diff, weight_diff, bias_diff]):
            print(f"\n✅ CORRECTNESS TEST PASSED! (tolerance: {tolerance})")
        else:
            print(f"\n⚠️  CORRECTNESS TEST FAILED! Some differences exceed tolerance {tolerance}")
            
    except Exception as e:
        print(f"❌ Correctness test failed: {str(e)}")


# memory_profile.py - Memory usage analysis
def profile_memory():
    """Profile memory usage"""
    print("="*50)
    print("MEMORY PROFILE")
    print("="*50)
    
    try:
        import psutil
        import os
        from dense import Dense as CythonDense
        
        process = psutil.Process(os.getpid())
        
        # Baseline memory
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
        print(f"Baseline memory: {baseline_memory:.1f} MB")
        
        # Create large layer
        large_layer = CythonDense(2048, 2048)
        after_creation = process.memory_info().rss / 1024 / 1024
        print(f"After creating 2048x2048 layer: {after_creation:.1f} MB")
        print(f"Layer memory usage: {after_creation - baseline_memory:.1f} MB")
        
        # Large batch forward pass
        large_input = np.random.randn(2048, 128).astype(np.float64)
        output = large_layer.forward(large_input)
        after_forward = process.memory_info().rss / 1024 / 1024
        print(f"After forward pass (batch=128): {after_forward:.1f} MB")
        
        # Backward pass
        output_grad = np.random.randn(2048, 128).astype(np.float64)
        input_grad = large_layer.backward(output_grad, 0.01)
        after_backward = process.memory_info().rss / 1024 / 1024
        print(f"After backward pass: {after_backward:.1f} MB")
        
    except ImportError:
        print("Install psutil for memory profiling: pip install psutil")
    except Exception as e:
        print(f"❌ Memory profiling failed: {str(e)}")


# main_test.py - Run all tests
def run_all_tests():
    """Run complete test suite"""
    print("🧪 CYTHON NEURAL NETWORK TEST SUITE")
    print("=" * 60)
    
    # 1. Compilation test
    if not test_compilation():
        print("\n❌ Compilation failed. Fix errors before proceeding.")
        return
    
    print("\n")
    
    # 2. Correctness test
    test_correctness()
    
    print("\n")
    
    # 3. Performance benchmark
    benchmark_performance()
    
    print("\n")
    
    # 4. Memory profile
    profile_memory()
    
    print("\n" + "="*60)
    print("🏁 TEST SUITE COMPLETE")


if __name__ == "__main__":
    run_all_tests()