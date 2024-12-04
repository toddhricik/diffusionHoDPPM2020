import sys
import numpy as np

import tensorflow.compat.v1 as tf
import tensorflow.contrib as tf_contrib
from tensorflow.python import debug as tf_debug

import diffusion_tf.models.unet as reverseDiffusionModel
import diffusion_tf.nn as 
import diffusion_tf.diffusion_utils as du 
from diffusion_tf import utils
import diffusion_tf.diffusion_utils_2 as du2

if __name__ == "__main__":
    # Set the number of timesteps between x_0 and x_0 in the latent space
    numTimesteps = 1000

    # Create a batch of ten (10) images 32x32 pixels each having 3 channels (R, G, B)
    # Each batch therefore has shape B x H x W x C
    fakeX = tf.zeros([10, 32, 32, 3])
    
    # Create a fake batch of y variables for labels
    fakeY = tf.zeros([10])
    
    # Define placeholders for inputs
    x_placeholder = tf.placeholder(tf.float32, shape=[None, fakeBatch.shape[-3], fakeBatch.shape[-2], fakeBatch.shape[-3]])
    y_placeholder = tf.placeholder(tf.float32, shape=[None, fakeBatch.shape[-3], fakeBatch.shape[-2], fakeBatch.shape[-3]])

    # We need a vector of randomly chosen time steps
    t = tf.random_uniform([10], 0, numTimesteps, dtype=tf.int32)
    
    # We need a vector of betas which should be an np.ndarray
    betas = du2.get_beta_schedule(beta_schedule='linear', beta_start=10e-4, beta_end=0.02, num_diffusion_timesteps=1000)
    
    # Create the model for the reverse diffusion process as described by Ho et al. 2020 DPPM paper (arxiv:2006.11239).
    reverseDiffusionModel = unet.Model(name="here", betas=betas, model_mean_type='eps', model_var_type='fixed', loss_type='mse', num_classes=1, dropout=0.1, randflip=1)

    # Define or create the loss function
    loss_fn = tf.square(tf.subtract(y_placeholder, estimated_targets))

    # Create a tensorflow 1.x session
    sess = tf.Session()

    # Set debbuging mode for session
    sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    
    # Set filter for has inf or nan
    sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)

    # Initialize the global variables 
    sess.run(tf.global_variables_initializer())

    # Demonstrate the time embedding with some tensorflow code here
    result = sess.run(get_time_embedding)    
