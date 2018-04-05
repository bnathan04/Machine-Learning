import tensorflow as tf
import numpy as np 

# build a layer in NN
def build_layer(input_layer, num_hidden_units):
    
    # initialize the weight matrix and bias vector
    num_inputs = input_layer.get_shape().as_list()[-1]
    W = tf.get_variable(name="Weights", shape=(num_inputs, num_hidden_units), dtype=tf.float64, initializer=tf.contrib.layers.xavier_initializer())
    b = tf.Variable(tf.zeros(shape=(1, num_hidden_units), dtype=tf.float64, name="Bias"))

    # input to next layer
    z = tf.add(tf.matmul(input_layer,W), b)

    return z
