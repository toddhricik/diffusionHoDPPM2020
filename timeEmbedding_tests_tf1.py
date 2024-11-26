import sys
import tensorflow.compat.v1 as tf
import tensorflow.contrib as tf_contrib

import diffusion_tf.models.unet as reverseDiffusionModel
import diffusion_tf.nn as 
import diffusion_tf.diffusion_utils as du 
from diffusion_tf import utils
import diffusion_tf.diffusion_utils_2 as du2



from tensorflow.python import debug as tf_debug

import numpy as np


if __name__ == "__main__":
    # ... your code ...
    sess = tf.Session()
    sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)

    # Run the graph with the debugger
    # sess = tf.Session(config=tfdbg.LocalCLIDebugWrapperSession.get_run_options())

    # Set the number of timesteps between x_0 and x_0 in the latent space
    numTimesteps = 1000

    # Create a batch of ten (10) images 32x32 pixels each having 3 channels (R, G, B)
    fakeBatch = tf.zeros([10, 32, 32, 3])

    # We need a vector of randomly chosen time steps
    t = tf.random_uniform([10], 0, numTimesteps, dtype=tf.int32)
    
    # We need a vector of betas which should be an np.ndarray
    betas = du2.get_beta_schedule(beta_schedule='linear', beta_start=10e-4, beta_end=0.02, num_diffusion_timesteps=1000)
    
    # Create the model for the reverse diffusion process as described by Ho et al. 2020 DPPM paper (arxiv:2006.11239).
    reverseDiffusionModel = unet.Model(name="here", betas=betas, model_mean_type='eps', model_var_type='fixed', loss_type='mse', num_classes=1, dropout=0.1, randflip=1)

    # Define or create the loss function

    # Define or create an optimizer

    ############ Training Algorithm 1 as described by Ho et al. 2020 DPPM paper (arxiv:2006.11239). ###########
    # Create a session
    with tf.Session() as sess:
        # Initialize variables
        # sess.run(tf.global_variables_initializer())
        # for epoch in range(num_epochs):
            # Your training loop
            
            # 1. Repeat
            # for batch in dataset:
            
                # 2. Sample x_0 from q           
                #    Get the independent and dependent variables from the batch
                # x_0, y = batch
                
                # 3. Sample t from uniform(1,...,T)
                #

                # 4. Sample epsilon from standard diagonal gaussian N(0, I)
                #    Compute epsilon_theta(( (sqrt(alpha_bar[t]) * x_0) + (sqrt(1-alpha_bar[t]) * epsilon), t) ))
                #    Note: The epsilon_theta((...)) mentioned above is the UNet whose forward process during traing takes x and t as parameters.
                #

                # 5. Take gradient descent step on ||epsilon - epsilon_theta(...)||^2    (See the paper above)
                #    gradients = optimizer.comput_gradients(loss)
                #    Apply the gradients
                # optimizer.apply_gradients(gradients)
                
                # Print the loss?
                #

        # 6. until converged

    # Enable tensorflow debugging (aka tfdbg)
    # sess.run(...)





