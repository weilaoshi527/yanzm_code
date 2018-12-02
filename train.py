#coding:utf-8

import tensorflow as tf
import os
import model.inference as model

TRAINING_STEPS = 30000
TRAINING_FOLDER = "model/train/*.tfrecords"
MODEL_SAVE_PATH = "model"
MODEL_NAME = "capcha.ckpt"
LEARNING_RATE = 0.001
KEEP_PROB = 0.75
BATCH_SIZE = 128
def next_batch(folder=TRAINING_FOLDER, batch_size=128):
    files = tf.train.match_filenames_once(folder)
    filename_queue = tf.train.string_input_producer(files, shuffle=False)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
            'label_index': tf.FixedLenFeature([], tf.string),
            'width': tf.FixedLenFeature([], tf.int64),
            'height': tf.FixedLenFeature([], tf.int64),
            'channels': tf.FixedLenFeature([], tf.int64),
            'raw_image': tf.FixedLenFeature([], tf.string)
        }
    )
    image = tf.decode_raw(features['raw_image'], tf.uint8)
    label = tf.decode_raw(features['label_index'], tf.float64)
    capacity = 1000 + 3 * batch_size
    image = tf.reshape(image, [model.INPUT_NODE,])
    image = tf.cast(image, tf.float32) / 255.0
    label = tf.reshape(label, [model.OUT_NODE,])
    image_batch, label_batch = tf.train.batch([image, label], batch_size, capacity=capacity)
    return image_batch, label_batch
if __name__ == '__main__':
    with tf.Graph().as_default() as g:
        X = tf.placeholder(tf.float32, [None, model.INPUT_NODE], name="x_input")
        Y = tf.placeholder(tf.float32, [None, model.OUT_NODE], name="y_input")
        keep_prob = tf.placeholder(tf.float32, name="keep_prob")
        output = model.inference(X, keep_prob=KEEP_PROB)
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=Y, logits=output))
        optm = tf.train.AdamOptimizer(learning_rate= LEARNING_RATE).minimize(loss)

        image_batch, label_batch = next_batch(batch_size=BATCH_SIZE)
        saver = tf.train.Saver()
        with tf.Session(config=tf.ConfigProto(device_count={"CPU":12},inter_op_parallelism_threads=1,intra_op_parallelism_threads=1,)) as sess:
            global_step = 0
            ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                global_step = ckpt.model_checkpoint_path.split(os.path.sep)[-1].split('-')[-1]
                global_step = (int)(global_step)
            else:
                sess.run((tf.global_variables_initializer(), tf.local_variables_initializer()))
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            for i in range(TRAINING_STEPS):
                cur_image_batch, cur_label_batch = sess.run([image_batch, label_batch])
                _, loss_ = sess.run([optm, loss], feed_dict={X: cur_image_batch, Y: cur_label_batch, keep_prob: KEEP_PROB})
                if i % 2000 == 0 and i > 0:
                    print("训练%d步后的loss值是%g." % (global_step, loss_))
                    saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step = global_step)
                global_step = global_step + 1
            coord.request_stop()
            coord.join(threads)