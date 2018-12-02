# coding=utf-8
import sys
import cv2
import Simple_Chars as s_char
import os
import common
import hashlib
import random
import argparse
import utils

FILE_EXT = ".png"
history_hash = []
def make_more_images(binary_image):
    images = []
    b_img = binary_image
    images.append(binary_image)
    e = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 2))
    images.append(cv2.erode(b_img.copy(), e, iterations=1))
    images.append(cv2.dilate(b_img.copy(), e, iterations=1))
    e = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
    images.append(cv2.erode(b_img.copy(), e, iterations=1))
    images.append(cv2.dilate(b_img.copy(), e, iterations=1))
    return images
def read_history(history_log):
    if os.path.exists(history_log):
        f = open(history_log)
        line = f.readline().strip()
        history_hash.append(line.strip())
        while (len(line)):
            line = f.readline().strip()
            history_hash.append(line.strip())
        f.close()
def sha1(file_name):
    fsize = os.path.getsize(file_name)
    f = open(file_name, 'rb')
    buffer = f.read(fsize)
    f.close()
    sha1_value = hashlib.sha1(buffer).hexdigest()
    return sha1_value
def in_history(file_name):
    sha1_value = sha1(file_name)
    if sha1_value in history_hash:
        return True
    return False
def update_history():
    f = open(history_log, 'w')
    for v in history_hash:
        f.write(v + "\n")
    f.close()
def line_count(chars):
    count = 1
    if len(chars) > 0:
        first = chars[0]
        x, y, w2, h2 = first.bbox
        for i in range(len(chars)):
            next = chars[i]
            if utils.greater_than_heavily(next.bbox[1], first.bbox[1]):
                count = count + 1
                first = next
    return count
def remove_first(chars):
    chars_ = []
    for char in chars:
        if char.bbox[0] > char.bbox[2]:
            chars_.append(char)
    return chars_
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-w", "--work_folder", required=False, help="需要处理的文件夹.")
    ap.add_argument("-t", "--train_folder", required=False, help="训练集文件夹.")
    ap.add_argument("-s", "--test_folder", required=False, help="测试集文件夹.")
    ap.add_argument("-e", "--error_folder", required=False, help="可能识别错误的文件夹.")
    ap.add_argument("-l", "--history", required=False, help="历史记录文件.")
    # ap.add_argument("-c", "--count", required=False, help="extracting count of tfRecords.")
    args = vars(ap.parse_args())
    dir_name = args["work_folder"]
    train_dir = args["train_folder"]
    test_dir = args["test_folder"]
    error_dir = args["error_folder"]
    history_log = args["history"]
    count = 100000
    read_history(history_log)
    for root, dirs, files in os.walk(dir_name):
        for file in files:
            name, ext = os.path.splitext(file)
            if ext == FILE_EXT:
                full_name = root + os.path.sep + file
                count = count + 1
                print(count - 100000, full_name)
                if in_history(full_name) == True:
                    continue
                history_hash.append(sha1(full_name))
                img = cv2.imread(full_name)
                family = s_char.Simple_Chars(img, "")
                chars = family.get_chars()
                if line_count(chars)==2:
                    chars = remove_first(chars)
                # common.crack_chars(chars)
                for char in chars:
                    images = make_more_images(char.get_image())
                    i = 1
                    test_seed = int(random.random()*len(images))
                    for b2_img in images:
                        save_dir = train_dir
                        if i == test_seed:
                            save_dir = test_dir
                        if char.probability <= 10.0:
                            save_dir = error_dir
                        png_name = save_dir + char.code + "_" + str(count) + "." + char.code + "_" + str(i) +".png"
                        cv2.imwrite(png_name, b2_img)
                        i = i + 1
    update_history()
