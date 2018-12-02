# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
import model.inference as model
import model.train as train
CHAR_SET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
CHAR_SET_LEN = len(CHAR_SET)


def vec2text(vec):
    char_pos = vec.nonzero()[0]
    text=[]
    for i, c in enumerate(char_pos):
        char_at_pos = i #c/63
        char_idx = c % CHAR_SET_LEN
        if char_idx < 10:
            char_code = char_idx + ord('0')
        elif char_idx < 36:
            char_code = char_idx - 10 + ord('A')
        elif char_idx < 62:
            char_code = char_idx-  36 + ord('a')
        elif char_idx == 62:
            char_code = ord('_')
        else:
            raise ValueError('error')
        text.append(chr(char_code))
    return "".join(text)
g = tf.Graph().as_default()
X = tf.placeholder(tf.float32, [None, model.INPUT_NODE], name="x_input")
keep_prob = tf.placeholder(tf.float32, name="keep_prob")
output = model.inference(X, keep_prob=train.KEEP_PROB)

predict = tf.argmax(tf.reshape(output, [-1, 1, model.OUT_NODE]), 2)
prob = tf.reshape(output, [-1, 1, model.OUT_NODE])
saver = tf.train.Saver()
sess = tf.Session()
ckpt = tf.train.get_checkpoint_state(train.MODEL_SAVE_PATH)
if ckpt and ckpt.model_checkpoint_path:
    saver.restore(sess, ckpt.model_checkpoint_path)
else:
    print("没有找到保存的模型.")

def crack_chars(number_array):
    for number in number_array:
        text_list, temp = sess.run([predict,prob], feed_dict={X: [number.get_image().flatten() / 255], keep_prob: 1})
        text = text_list[0].tolist()
        vector = np.zeros(model.OUT_NODE)
        i = 0
        for n in text:
            vector[i * model.OUT_NODE + n] = 1
            i += 1
        number.code = vec2text(vector)
        number.prob = np.amax(temp)
    return number_array
def crack_char(binary_image):
    text_list, temp = sess.run([predict, prob], feed_dict={X: [binary_image.flatten() / 255], keep_prob: 1})
    text = text_list[0].tolist()
    vector = np.zeros(model.OUT_NODE)
    i = 0
    for n in text:
        vector[i * model.OUT_NODE + n] = 1
        i += 1
    code = vec2text(vector)
    probability = np.amax(temp)
    return code, probability
def crack_binary_chars(binary_images):
    result = []
    for b_img in binary_images:
        code, probability = crack_char(b_img)
        result.append((code, probability))
    return result