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

def line_count(chars):
    count = 0
    if len(chars) > 0:
        point_y = []
        for i in range(len(chars)):
            x, y, w, h = chars[i].bbox
            y = (int)(y / 10)
            point_y.append(y)
        s_list = utils._statistics(point_y)
        for item, item_count in s_list:
            if item_count > 2:
                count = count + 1
    return count
def remove_first(chars):
    chars_ = []
    for char in chars:
        if char.bbox[0] > 10:
            chars_.append(char)
    return chars_
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-w", "--work_folder", required=False, help="需要处理的文件夹.")
    args = vars(ap.parse_args())
    dir_name = args["work_folder"]
    for root, dirs, files in os.walk(dir_name):
        for file in files:
            name, ext = os.path.splitext(file)
            if ext == FILE_EXT:
                full_name = root + os.path.sep + file
                img = cv2.imread(full_name)
                family = s_char.Simple_Chars(img, "")
                chars = family.get_chars()
                chars_line_count = family.get_line_count()
                if chars_line_count == 2:
                    chars = remove_first(chars)
                # common.crack_chars(chars)
                char_codes = ""
                for char in chars:
                    char_codes = char_codes + char.code
                start = 0
                interval = (int)(len(char_codes) / chars_line_count)
                new_name = ""
                for i in range(chars_line_count):
                    new_name = new_name + char_codes[start:start + interval] + "."
                    start = start + interval
                new_name = root + os.sep + new_name[:-1] + FILE_EXT
                if os.path.exists(new_name) and full_name != new_name:
                    os.remove(new_name)
                if full_name != new_name:
                    os.rename(full_name, new_name)
                print(full_name + ":  -->  " + new_name)

