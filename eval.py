#coding:utf-8

import tensorflow as tf
import os
import time
import model.inference as model
import model.train as train

EVAL_FOLDER = "model/test/*.tfrecords"
EVAL_INTERVAL_SECS = 120
def evaluate():
    with tf.Graph().as_default() as g:
        X = tf.placeholder(tf.float32, [None, model.INPUT_NODE], name="x_input")
        Y = tf.placeholder(tf.float32, [None, model.OUT_NODE], name="y_input")
        keep_prob = tf.placeholder(tf.float32, name="keep_prob")
        output = model.inference(X, keep_prob=train.KEEP_PROB)

        predict = tf.reshape(output, [-1, 1, model.OUT_NODE])
        max_idx_p = tf.arg_max(predict, 2)
        max_idx_l = tf.arg_max(tf.reshape(Y, [-1, 1, model.OUT_NODE]), 2)
        accuracy = tf.reduce_mean(tf.cast(tf.equal(max_idx_p, max_idx_l), tf.float32))
        image_batch, label_batch = train.next_batch(folder=EVAL_FOLDER, batch_size=1000)
        saver = tf.train.Saver()
        while True:
            with tf.Session() as sess:
                ckpt = tf.train.get_checkpoint_state(train.MODEL_SAVE_PATH)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    global_step = ckpt.model_checkpoint_path.split(os.path.sep)[-1].split('-')[-1]
                else:
                    print("没有找到保存的模型.")
                    return
                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(sess=sess, coord=coord)
                cur_image_batch, cur_label_batch = sess.run([image_batch, label_batch])
                accuracy_score = sess.run(accuracy, feed_dict={X: cur_image_batch, Y: cur_label_batch, keep_prob: train.KEEP_PROB})
                print("训练%s步后的概率值是%g." % (global_step, accuracy_score))
                coord.request_stop()
                coord.join(threads)
            time.sleep(EVAL_INTERVAL_SECS)
if __name__ == '__main__':
    evaluate()