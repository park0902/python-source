from tensorflow.python.training import moving_averages as ema
import tensorflow as tf

def batch_norm(self, input, shape, training, convl=True, name='BN', decay=0.99):
    beta = tf.Variable(tf.constant(0.0, shape=[shape]), name='beta')
    scale = tf.Variable(tf.constant(1.0, shape=[shape]), name='scale')
    moving_collections = ['moving_variables', tf.GraphKeys.MOVING_AVERAGE_VARIABLES]
    moving_mean = tf.Variable(tf.zeros([shape]), trainable=False, collections=moving_collections, name='moving_mean')
    moving_var = tf.Variable(tf.ones([shape]), trainable=False, collections=moving_collections, name='moving_var')

    if training is True:
        if convl:
            mean, var = tf.nn.moments(input, [0, 1, 2], name='moments')
        else:
            mean, var = tf.nn.moments(input, [0], name='moments')

        update_moving_mean = ema.assign_moving_average(moving_mean, mean, decay)
        tf.add_to_collection('_update_ops_', update_moving_mean)
        update_moving_var = ema.assign_moving_average(moving_var, var, decay)
        tf.add_to_collection('_update_ops_', update_moving_var)
        tf.assign_sub()
    else:
        mean, var = moving_mean, moving_var

    output = tf.nn.batch_normalization(input, mean, var, beta, scale, variance_epsilon=1e-3, name=name)
    output.set_shape(input.get_shape())
    return output







from tensorflow.python.training import moving_averages as ema
import tensorflow as tf

def batch_norm(self, input, shape, training, convl=True, name='BN', decay=0.99):
    beta = tf.Variable(tf.constant(0.0, shape=[shape]), name='beta')
    scale = tf.Variable(tf.constant(1.0, shape=[shape]), name='scale')
    moving_collections = ['moving_variables', tf.GraphKeys.MOVING_AVERAGE_VARIABLES]
    moving_mean = tf.Variable(tf.zeros([shape]), trainable=False, collections=moving_collections, name='moving_mean')
    moving_var = tf.Variable(tf.ones([shape]), trainable=False, collections=moving_collections, name='moving_var')

    if training is True:
        if convl:
            mean, var = tf.nn.moments(input, [0, 1, 2], name='moments')
        else:
            mean, var = tf.nn.moments(input, [0], name='moments')

        update_moving_mean = tf.assign_sub(moving_mean, mean)
        tf.add_to_collection('_update_ops_', update_moving_mean)
        update_moving_var = tf.assign_sub(moving_var, var)
        tf.add_to_collection('_update_ops_', update_moving_var)
        tf.assign_sub()
    else:
        mean, var = moving_mean, moving_var

    output = tf.nn.batch_normalization(input, mean, var, beta, scale, variance_epsilon=1e-3, name=name)
    output.set_shape(input.get_shape())
    return output