import tensorflow.compat.v1 as tf
import tensorflow.contrib as tf_contrib

import diffusion_tf.models.unet as unet
import diffusion_tf.residualBlocks as rb



if __name__ == "__main__":

    # Instantiate a fake batch of images all elements set to zero
    fakeBatch_x = tf.zeros([10, 32, 32, 3])

    # 
