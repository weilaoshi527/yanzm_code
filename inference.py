#coding:utf-8

import tensorflow as tf
INPUT_NODE = 256
INPUT_SHAPE = [-1, 16, 16, 1]
LAYER1_NODE = 32
LAYER2_NODE = 64
LAYER3_NODE = 64
LAYER5_NODE = 1024
PADDING = "SAME"
C_STRIDES = [1, 1, 1, 1]
P_STRIDES = [1, 2, 2, 1]
P_SIZE = [1, 2, 2, 1]
OUT_NODE = 36

def get_weight_variable(shape):
    return tf.Variable(0.01 * tf.random_normal(shape))
def get_bias_variable(layer_node):
    return tf.Variable(0.1 * tf.random_normal([layer_node]))
def conn_layer(input, last_size, cur_size, keep_prob, layer_name):
    with tf.variable_scope(layer_name):
        weight = get_weight_variable([3, 3, last_size, cur_size])
        bias = get_bias_variable(cur_size)
        out = tf.nn.conv2d(input, weight, strides = C_STRIDES, padding = PADDING)
        out = tf.nn.bias_add(out, bias)
        out = tf.nn.relu(out)
        out = tf.nn.max_pool(out, ksize = P_SIZE, strides = P_STRIDES, padding = PADDING)
        out = tf.nn.dropout(out, keep_prob = keep_prob)
    return out
def fc_layer(input, last_size, cur_size, layer_name):
    with tf.variable_scope(layer_name):
        weight = get_weight_variable([last_size, cur_size])
        bias = get_bias_variable(cur_size)
        fc = tf.nn.bias_add(tf.matmul(input, weight), bias)
    return fc
def inference(input_tensor, keep_prob=.75):
    x = tf.reshape(input_tensor, shape = INPUT_SHAPE)
    out = conn_layer(x, 1, LAYER1_NODE, keep_prob, "layer1")
    out = conn_layer(out, LAYER1_NODE, LAYER2_NODE, keep_prob, "layer2")
    out = conn_layer(out, LAYER2_NODE, LAYER3_NODE, keep_prob, "layer3")
    shape = out.get_shape().as_list()
    dim = 1
    for d in shape[1:]:
        dim *= d
    out = tf.reshape(out, [-1, dim])
    out = fc_layer(out, dim, LAYER5_NODE, "layer4")
    out = tf.nn.relu(out)
    out = tf.nn.dropout(out, keep_prob)
    out = fc_layer(out, LAYER5_NODE, OUT_NODE, "layer5")
    return out