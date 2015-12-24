import tensorflow as tf
import numpy as np


def conv_relu(input, kernel_shape, bias_shape):
    # Create variable named "weights".
    weights = tf.get_variable("weights", kernel_shape,
                              initializer=tf.random_normal_initializer())
    # Create variable named "biases".
    biases = tf.get_variable("biases", bias_shape,
                             initializer=tf.constant_initializer())
    conv = tf.nn.conv2d(input, weights,
                        strides=[1, 1, 1, 1], padding='SAME')
    return tf.nn.relu(conv + biases)

def my_image_filter(input_images):
    with tf.variable_scope("conv1"):
        # Variables created here will be named "conv1/weights", "conv1/biases".
        relu1 = conv_relu(input_images, [5, 5, 32, 32], [32])

    with tf.variable_scope("conv2"):
        # Variables created here will be named "conv2/weights", "conv2/biases".
        return conv_relu(relu1, [5, 5, 32, 32], [32])


if __name__ == '__main__':

    image1 = np.random.rand(5, 5, 32, 32)
    image1 = tf.Variable(image1.astype(np.float32))
    image2 = np.random.rand(5, 5, 32, 32)
    image2 = tf.Variable(image2.astype(np.float32))
        
    try:
        result1 = my_image_filter(image1)
        result2 = my_image_filter(image2)
    except Exception as e:
        print e

        
    with tf.variable_scope("image_filters") as scope:
        result1 = my_image_filter(image1)
        scope.reuse_variables()
        result2 = my_image_filter(image2)

        print result2

    init_op = tf.initialize_all_variables()
    with tf.Session() as sess:
        # init
        sess.run(init_op)
        
        sess.run(result2)
        print result2.eval()
        
