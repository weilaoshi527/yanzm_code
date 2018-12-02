# coding=utf-8
import sys
import cv2
import Line_Chars as l_char
import Simple_Chars as s_char
import Rect_Chars as r_char
import semantic_parser as sp
import urllib.parse as parse
import base64
import numpy as np
import utils
import os
def unpack(image_content, semantic):
    base64_string = parse.unquote(image_content)
    pix = np.fromstring(base64.b64decode(base64_string), np.uint8)
    img = cv2.imdecode(pix, 3)
    semantic_decode = parse.unquote(semantic)
    return get_result(img, semantic_decode)
def get_result(img, semantic):
    s = sp.Semantic_parser(semantic)
    family = None
    if s.is_line:
        family = l_char.Line_Chars(img, semantic)
    else:
        if s.is_rect:
            family = r_char.Frame_Chars(img, semantic)
        else:
            family = s_char.Simple_Chars(img, semantic)
    return family.get_result()
if __name__ == '__main__':
    file_name = sys.argv[1]
    if len(sys.argv) > 2:
        semantic = sys.argv[2]
        if len(sys.argv) > 3:
            utils.DEBUG = True
            utils.DEBUG_DIR = os.path.splitext(file_name)[0] + os.sep
    else:
        utils.DEBUG = True
        utils.DEBUG_DIR = os.path.splitext(file_name)[0] + os.sep
        semantic = ""
    img = cv2.imread(file_name)
    print(get_result(img, semantic))