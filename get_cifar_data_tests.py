import sys
import os
import numpy as np

import tensorflow.compat.v1 as tf
import tensorflow.contrib as tf_contrib
from tensorflow.python import debug as tf_debug




if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data() 
    print("Shape of training images:", x_train.shape)  
    assert x_train.shape == (50000, 32, 32, 3)
    print("Shape of training labels:", y_train.shape) 
    assert y_train.shape == (50000, 1)
    print("Shape of testing images:", x_test.shape) 
    assert x_test.shape == (10000, 32, 32, 3)
    print("Shape of testing labels:", y_test.shape)
    assert y_test.shape == (10000, 1)
