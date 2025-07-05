import numpy as np
print("NumPy version:", np.__version__)

try:
    import tensorflow as tf
    print("TensorFlow version:", tf.__version__)
    print("GPU Available:", tf.config.list_physical_devices('GPU'))
    
    # Test TensorFlow operations
    a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
    b = tf.constant([[5.0, 6.0], [7.0, 8.0]])
    c = tf.matmul(a, b)
    print("Test matrix multiplication result:")
    print(c)
    
    print("TensorFlow installation successful!")
except Exception as e:
    print("Error testing TensorFlow:", e)
