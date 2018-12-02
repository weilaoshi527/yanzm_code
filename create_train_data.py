# encoding: utf-8
import cv2
import numpy as np
import sys
import tensorflow as tf
import glob
import os
import random
import argparse

labels = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
img_width = 16
img_height = 16
char_count = len(labels)
def _rotate(binary_imgs, count=3):
    imgs = []
    k = count
    for img in binary_imgs:
        for i in range(0 - k, k + 1):
            M = cv2.getRotationMatrix2D((img_width, img_height), i, 1.0)
            imgs.append(cv2.warpAffine(img,M,(img_width,img_height)))
    return imgs
def _clip(binary_imgs, count=1):
    imgs = []
    k = count
    for img in binary_imgs:
        for i in range(0 - k, k + 1):
            for j in range(0 - k, k + 1):
                H = np.float32([[1, 0, i], [0, 1, j]])
                imgs.append(cv2.warpAffine(img, H, (img_width, img_height)))
    return imgs
def _add_line_noise(binary_imgs, count=12):
    imgs = []
    k = count
    for img in binary_imgs:
        imgs.append(img)
        for i in range(k):
            p1 = int(np.random.rand() * img_width * .75), int(np.random.rand() * img_height * .75)
            p2 = int(np.random.rand() * img_width * .75), int(np.random.rand() * img_height * .75)
            img2 = img.copy()
            imgs.append(cv2.line(img2, p1, p2, 255, 1))
    return imgs
def _erode_and_dilate(binary_imgs, count=2):
    imgs = []

    k = count
    for img in binary_imgs:
        imgs.append(img)
        e = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 2))
        imgs.append(cv2.erode(img.copy(), e, iterations=1))
        imgs.append(cv2.dilate(img.copy(), e, iterations=1))
        e = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
        imgs.append(cv2.erode(img.copy(), e, iterations=1))
        imgs.append(cv2.dilate(img.copy(), e, iterations=1))
        e = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
        imgs.append(cv2.erode(img.copy(), e, iterations=1))
        imgs.append(cv2.dilate(img.copy(), e, iterations=1))
    return imgs
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))
def files_to_tfRecords(files_glob, output):
    file_names = glob.glob(files_glob)
    raw_images = []
    label_indexs = []
    for file_name in file_names:
        gray = cv2.imread(file_name)[:, :, 0]
        # _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        code = os.path.split(file_name)[1][:1]
        label = np.zeros(char_count)
        label[labels.index(code)] = 1.0
        raw_images.append(gray)
        label_indexs.append(label)
    if output == None:
        output = os.path.split(files_glob)[0] + ".tfrecords"
    write_to_tfRecords(output, raw_images, label_indexs)
def write_to_tfRecords(file_name, images, labels):
    writer = tf.python_io.TFRecordWriter(file_name)
    shuffled = list(zip(images,labels))
    random.shuffle(shuffled)
    for i in range(len(shuffled)):
        raw_image = shuffled[i][0].reshape(img_width * img_height).tostring()
        label = shuffled[i][1].tostring()
        tfrecord = tf.train.Example(features=tf.train.Features(feature={
            'label_index': _bytes_feature(label),
            'height': _int64_feature(img_height),
            'width': _int64_feature(img_width),
            'channels': _int64_feature(1),
            'raw_image': _bytes_feature(raw_image)
        }))
        writer.write(tfrecord.SerializeToString())
    writer.close()
def tfRecords_to_files(tfRecords_file, folder, extract_count = 100):
    reader = tf.TFRecordReader()
    file_queue = tf.train.string_input_producer([tfRecords_file])
    _, serialized_tfRecord = reader.read(file_queue)
    features = tf.parse_single_example(
        serialized_tfRecord,
        features={
            'label_index': tf.FixedLenFeature([], tf.string),
            'width': tf.FixedLenFeature([], tf.int64),
            'height': tf.FixedLenFeature([], tf.int64),
            'channels': tf.FixedLenFeature([], tf.int64),
            'raw_image': tf.FixedLenFeature([], tf.string)
        })

    image = tf.decode_raw(features['raw_image'], tf.uint8)
    image = tf.reshape(image, [img_width, img_height])
    label = tf.decode_raw(features['label_index'], tf.float64)
    sess = tf.Session()
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    for i in range(extract_count):
        x, y = sess.run([image, label])
        image_file = folder + os.sep + labels[np.argmax(y)] + "_" + str(i) + ".png"
        cv2.imwrite(image_file, x)
        print(image_file)
    coord.request_stop()
    coord.join(threads)
    sess.close()
def create_tfRecords(file_name, output, tfrecord_count= 2000):
    img = cv2.imread(file_name)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
    _, contours, hierarchy = cv2.findContours(binary.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    chars = []
    labels = []
    contours.sort(key=lambda y:cv2.boundingRect(y)[0])
    i = 0
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        b_img = binary[y: y + h, x: x + w]
        b_img = _resize(b_img, img_width)
        change_imgs = []
        change_imgs.append(b_img)
        change_imgs = _add_line_noise(change_imgs, count = int(tfrecord_count / 7))  #*6
        change_imgs = _erode_and_dilate(change_imgs)    #*7
        label = np.zeros(char_count)
        label[i] = 1.0
        for b_img in change_imgs:
            chars.append(b_img)
            labels.append(label)
        i = i + 1
    if output == None:
        output = file_name + ".tfRecords"
    write_to_tfRecords(output, chars, labels)
def _resize(img, width):
    h, w = img.shape
    radio = width / max(w, h)
    h2, w2 = (int)(h * radio), (int)(w * radio)
    img2 = cv2.resize(img, (w2, h2))
    img3 = np.zeros((width, width), np.uint8)
    if h2 < w2:
        img3[:h2, :] = img2
    else:
        img3[:, :w2] = img2
    return img3
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    # ap.add_argument("-i", "--image", required=False, help="带有一行字母的图片.")
    # ap.add_argument("-o", "--output", required=False, help="tfRecords训练文件.")
    ap.add_argument("-f", "--from", required=False, help="导入文件，形如/dir/*.png.")
    ap.add_argument("-t", "--to", required=False, help="导出文件.")
    ap.add_argument("-c", "--count", required=False, help="extracting count of tfRecords.")
    args = vars(ap.parse_args())
    # file_name = args["image"]
    # output_tfRecord_file = args["output"]
    from_file = args["from"]
    to_file = args["to"]
    count = args["count"]
    # if file_name != None:
    #     create_tfRecords(file_name, output_tfRecord_file)
    if from_file != None:
        if os.path.isfile(from_file):
            if count == None:
                count = 50
            tfRecords_to_files(from_file, to_file, extract_count = (int)(count))
        else:
            files_to_tfRecords(from_file, to_file)
    print(from_file, to_file)

