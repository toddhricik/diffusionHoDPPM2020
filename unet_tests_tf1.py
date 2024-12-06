import sys
import os
import numpy as np

import tensorflow.compat.v1 as tf
import tensorflow.contrib as tf_contrib
from tensorflow.python import debug as tf_debug

import diffusion_tf.models as models 
import diffusion_tf.nn as nn 
import diffusion_tf.diffusion_utils as du 
from diffusion_tf import utils
import diffusion_tf.diffusion_utils_2 as du2

from diffusion_tf.diffusion_utils_2 import get_beta_schedule, GaussianDiffusion2
import diffusion_tf.models.unet as unet


if __name__ == "__main__":

    # Create a tensorflow 1.x session
    with tf.Session() as sess:

    # Set debbuging mode for tfdbg cli session
    sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    
    # Set filter for has inf or nan
    sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)    # Set the number of timesteps between x_0 and x_0 in the latent space

    # Set the number of time steps used in the forward diffusion
    numTimesteps = 1000

    # Create a batch of ten (10) images 32x32 pixels each having 3 channels (R, G, B)
    # Each batch therefore has shape B x H x W x C
    fakeX = tf.zeros([10, 32, 32, 3])
    
    # Create a fake batch of y variables for labels
    fakeY = tf.zeros([10], dtype=tf.int32)
    
    # Define placeholders for inputs
    x_placeholder = tf.placeholder(tf.float32, shape=[None, fakeX.shape[-3], fakeX.shape[-2], fakeX.shape[-1]])
    y_placeholder = tf.placeholder(tf.float32, shape=[None, 1])

    # We need a vector of randomly chosen time steps
    t = tf.random_uniform([10], 0, numTimesteps, dtype=tf.int32)
    
    # We need a vector of betas which should be an np.ndarray
    betas = du2.get_beta_schedule(beta_schedule='linear', beta_start=10e-4, beta_end=0.02, num_diffusion_timesteps=1000)
    
    # Set thenumber of classes
    num_classes = 1 # tf.size(tf.unique(fakeY))

    
    # Create a diffusion model
    diffusionModel = du2.GaussianDiffusion2(betas=betas, model_mean_type='eps', model_var_type='fixed', loss_type='mse')
    # Create the model for the reverse diffusion process as described by Ho et al. 2020 DPPM paper (arxiv:2006.11239).
    result = unet.model(x=fakeX, t=t, y=None, name="here", num_classes=num_classes, ch=128, ch_mult=(1,2,2,2), num_res_blocks=2, attn_resolutions=(16,), out_ch=256, dropout=0., resamp_with_conv=True)

    # Define or create the loss function
    loss_fn = tf.square(tf.subtract(y_placeholder, result))


    # Initialize the global variables 
    sess.run(tf.global_variables_initializer())

    # Demonstrate the time embedding with some tensorflow code here
    result = sess.run(get_time_embedding)    
