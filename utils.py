# coding=utf-8
import numpy as np
import cv2
import os
import char

DEBUG = False
DEBUG_DIR = "images" + os.sep + "debug" + os.sep

LINE_MAX_WIDTH = 3
CHAR_MIN_RADIUS = 6.5
CHAR_MAX_RADIUS = 14.5
CHAR_MIN_WIDTH = 2
CHAR_MAX_WIDTH = 24
CHAR_MIN_HEIGHT = 10
CHAR_MAX_HEIGHT = 30

class Color:
    def __init__(self, rgb, name):
        self.rgbs = []
        self.rgbs.append(rgb)
        self.name = name
    def append(self, rgb):
        if rgb not in self.rgbs:
            self.rgbs.append(rgb)
def init_colors():
    Colors = []
    black = Color((0, 0, 0), "黑")
    Colors.append(black)
    red = Color((255, 0, 0), "红")
    red.append((255, 128, 128))
    Colors.append(red)
    green = Color((0, 255, 0), "绿")
    green.append((128, 255, 0))
    green.append((128, 255, 128))
    green.append((0, 255, 128))
    green.append((0, 255, 64))
    green.append((0, 128, 0))
    green.append((0, 128, 64))
    Colors.append(green)
    blue = Color((0, 0, 255), "蓝")
    blue.append((0, 128, 255))
    blue.append((0, 128, 192))
    blue.append((0, 0, 160))
    blue.append((0, 153, 255))
    Colors.append(blue)
    yellow = Color((255, 255, 0), "黄")
    yellow.append((255, 255, 128))
    yellow.append((128, 128, 0))
    yellow.append((255, 165, 0))
    yellow.append((255, 204, 0))
    Colors.append(yellow)
    violet = Color((255, 0, 255), "紫")
    violet.append((128, 0, 128))
    violet.append((128, 0, 255))
    violet.append((255, 128, 255))
    Colors.append(violet)
    return Colors
def greater_than_heavily(x, y, delta=1):
    if x > delta * y + y:
        return True
    return False
# def smaller_than_heavily(x, y):
#     if y > 2 * x:
#         return True
#     return False
def approximate(x, y, delta=0.25):
    a = min(x, y)
    b = max(x, y)
    if b > a * (1 - delta) and b < a * (1 + delta):
        return True
    return False
def bgr_to_one(img):
    B, G, R = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    new_img = (np.uint32(R)<<16) + (np.uint32(G)<<8)  + B
    return new_img
# def one_to_rgb(color_value):
#     color_value = np.uint32(color_value)
#     B = np.uint32(color_value<<24)>>24
#     G = np.uint32(color_value<<16)>>24
#     R = np.uint32(color_value<<8)>>24
#     return R, G, B
def top_k(one_img):
    pixels = statistics(one_img)
    img_list = []
    for (rgb, count) in pixels:
        img2 = one_img.copy() - rgb
        img3 = np.where(img2 < 256 , 255, 0)  #这个地方有问题?
        img_list.append(np.uint8(img3))
    return img_list
def statistics(one_img, filter_min_count=25):
    x, y = one_img.shape
    h = {}
    for i in range(x):
        for j in range(y):
            p = one_img[i, j]
            if p != 16777215:  # except white color 255,255,255
                if p in h.keys():
                    h[p] = h[p] + 1
                else:
                    h[p] = 0
    pixel_list = []
    for rgb, count in h.items():
        if count > filter_min_count:
            pixel_list.append((rgb, count))
    pixel_list.sort(key=lambda y: y[0])
    return pixel_list

def analyze_image_color(img):
    result = "unknown_color"
    colors = init_colors()
    B, G, R = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    one_img = (np.uint32(R) << 16) + (np.uint32(G) << 8) + B
    x, y = one_img.shape
    h = {}
    for i in range(x):
        for j in range(y):
            p = one_img[i, j]
            if p != 16777215:  # except white color 255,255,255
                if p in h.keys():
                    h[p] = h[p] + 1
                else:
                    h[p] = 0
    pixels = []
    for k, v in h.items():
        pixels.append((k, v))
    pixels.sort(key=lambda y: y[len(y) - 1])

    (color_value, color_count) = pixels[len(pixels) - 1]
    color_value = np.uint32(color_value)
    b = np.uint32(color_value << 24) >> 24
    g = np.uint32(color_value << 16) >> 24
    r = np.uint32(color_value << 8) >> 24
    distance = 99999999
    for c in colors:
        for (r2, g2, b2) in c.rgbs:
            r3 = (r - r2) * (r - r2)
            g3 = (g - g2) * (g - g2)
            b3 = (b - b2) * (b - b2)
            distance2 = r3 + g3 + b3
            if distance2 < distance:
                result = c.name
                distance = distance2
    return result
def write_debug(img_dir, img, debug_index):
    if os.path.exists(img_dir) != True:
        os.mkdir(img_dir)
    cv2.imwrite(img_dir + os.path.sep + str(debug_index) + ".png", img)
    debug_index = debug_index + 1
    return debug_index

# def _is_a_line(binary_image):
#     h_sum = np.sum(binary_image, axis=0)
#     v_sum = np.sum(binary_image, axis=1)
#     peaks = _get_peaks(h_sum)
#     if len(peaks) == 1:
#         if peaks[0] == 255:
#             return True
#     peaks = _get_peaks(v_sum)
#     if len(peaks) == 1:
#         if peaks[0] == 255:
#             return True
def _get_peaks(sum):
    peaks = []
    start = sum[0]
    if start > 0:
        peaks.append(start)
    for s in sum:
        if s > 0:
            if s != start:
                peaks.append(s)
                start = s
    return peaks
# def _is_a_rectangle(binary_image):
#     h_sum = np.sum(binary_image, axis=0)
#     v_sum = np.sum(binary_image, axis=1)
#     peaks = _get_peaks(h_sum)
#     if len(peaks) == 3:
#         if peaks[0] == peaks[2]:
#             return True
#     peaks = _get_peaks(v_sum)
#     if len(peaks) == 3:
#         if peaks[0] == peaks[2]:
#             return True
# def _split_chars(h_sum, b_img):
#     for i in range(len(h_sum)):
#         if h_sum[i] <= 255 * 2:
#             b_img[:, i] = 0
#     return b_img
def _is_majority_height(height, majority_height, delta=3):
    if height > majority_height - delta and height <= majority_height + delta:
        return True
    return False
def _remove_smaller_chars(faked_chars):
    majority_height = _majority_height(faked_chars)
    min_height = 10000
    i = 0
    min_index = -1
    for c in faked_chars:
        char_img = c.b_img
        height = char_img.shape[0]
        if height < min_height:
            min_height = height
            min_index = i
        i = i + 1
    if min_index > -1:
        c = faked_chars[min_index]
        height = c.bbox[3]
        if _is_majority_height(height, majority_height) == False:
            faked_chars.remove(c)
            return _remove_smaller_chars(faked_chars)
    return faked_chars
def _copy_to(from_img, copy_rect, to_image_shape):
    back_image = np.zeros(to_image_shape, np.uint8)
    x, y, w, h = copy_rect
    back_image[y:y + h, x:x + w] = from_img
    return back_image
def _remove_chinese_chars(faked_chars, shape):
    removed_chars = []
    for char in faked_chars:
        if char.confirmed_code() == "":
            if _is_chinese_code(char.b_img):
                removed_chars.append(char)
    for char in removed_chars:
        faked_chars.remove(char)
    return faked_chars
def _majority_height(faked_chars):
    heights = {}
    for char in faked_chars:
        height, width = char.b_img.shape
        if height in heights.keys():
            heights[height] = heights[height] + 1
        else:
            heights[height] = 0
    heights_list = []
    for height, count in heights.items():
        heights_list.append((height, count))
    heights_list.sort(key=lambda y: y[1])
    max_height = heights_list[-1]
    majority_height = 0
    count = 0
    for i in range(len(heights_list)):
        if heights_list[0 - i - 1][1] == max_height[1]:
            majority_height = majority_height + heights_list[0 - i - 1][0]
            count = count + 1
    return majority_height / count
def _remove_bounding_box(img, bounding_width=2):
    img2 = img.copy()
    for i in range(3):
        img2[:, :, i][:bounding_width, :] = 255
        img2[:, :, i][:, :bounding_width] = 255
        img2[:, :, i][-bounding_width:, :] = 255
        img2[:, :, i][:, -bounding_width:] = 255
    return img2
# def _remove_bigger_chars(faked_chars, shape):
#     i = -1
#     majority_height = _majority_height(faked_chars)
#     for char in faked_chars:
#         key, char_img, c = char
#         x, y, w, h = cv2.boundingRect(c)
#         i = i + 1
#         if _is_majority_height(h, majority_height, delta=2):
#             continue
#         back_image = _copy_to(char_img, (x, y, w, h), shape)
#         left = _remove_small_line_block(back_image, c, majority_height)
#         if left == None:
#             faked_chars.remove(char)
#             return _remove_bigger_chars(faked_chars, shape)
#         else:
#             faked_chars[i] = left
#     return faked_chars
def _remove_small_line_block(back_image, contour, majority_height):
    x, y, w, h = cv2.boundingRect(contour)
    new_image = np.zeros(back_image.shape, np.uint8)
    new_image[y:y + h, x:x + w] = back_image[y:y + h, x:x + w]
    h_sum = np.sum(new_image, axis=0)
    for i in range(len(h_sum)):
        if h_sum[i] == 255:
            new_image[:, i] = 0
    num, labels, stat, _ = cv2.connectedComponentsWithStats(new_image,connectivity=4)
    if len(stat) > 1:
        max_labels = np.argmax(stat[1:, 4]) + 1 #除了背景以外的最多4联通像素
        new_img = np.uint8(np.where(labels == max_labels, 255, 0))
    else:
        new_img = new_image
    _, contours, hierarchy = cv2.findContours(new_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        x2, y2, w2, h2 = cv2.boundingRect(c)
        if w2 > CHAR_MIN_WIDTH and w2 <= CHAR_MAX_WIDTH and h2 > CHAR_MIN_HEIGHT and h2 <= CHAR_MAX_HEIGHT:
            if _is_majority_height(h2, majority_height):
                order_key = int(y2 / 10) * 1000 + x2
                return (order_key, back_image[y2:y2 + h2, x2:x2 + w2], c)
    return None
def contains(big_rect, small_rect):
    x, y, w, h = big_rect
    x2, y2, w2, h2 = small_rect
    if x2 >= x and x2 + w2 <= x + w and y2 >= y and y2 + h2 <= y + h:
        return True
    return False

# def _split_two_chars(back_image, contour):
#     x, y, w, h = cv2.boundingRect(contour)
#     new_image = np.zeros(back_image.shape, np.uint8)
#     new_image[y:y + h, x:x + w] = back_image[y:y + h, x:x + w]
#     if _is_a_line(new_image) or _is_a_rectangle(new_image):
#         return np.zeros(back_image.shape, np.uint8)
#     if approximate(w, back_image.shape[1]) or approximate(h, back_image.shape[0]):
#         return np.zeros(back_image.shape, np.uint8)
#     h_sum = np.sum(new_image, axis=0)
#     v_sum = np.sum(new_image, axis=1)
#     if w > CHAR_MAX_WIDTH:
#         for i in range(len(h_sum)):
#             if h_sum[i] <= 255:
#                 new_image[:, i] = 0
#     if h > CHAR_MAX_HEIGHT:
#         for i in range(len(v_sum)):
#             if v_sum[i] <= 255:
#                 new_image[i, :] = 0
#     return new_image
def _statistics(s_list):
    s_dict = {}
    for s in s_list:
        if s in s_dict.keys():
            s_dict[s] = s_dict[s] + 1
        else:
            s_dict[s] = 1
    s_ordered = []
    for s, count in s_dict.items():
        s_ordered.append((s, count))
    s_ordered.sort(key=lambda y: y[1])
    return s_ordered
def _max_frequency(s_list):
    s_ordered = _statistics(s_list)
    if len(s_ordered)>0:
        return s_ordered[-1]
    return 0, 0
def _remove_line(binary_image):
    num, labels, stat, _ = cv2.connectedComponentsWithStats(binary_image, connectivity=4)
    for i in range(num):
        if stat[i, 4] <= 4:
            labels = np.uint8(np.where(labels == i, 0, labels))
    labels = np.uint8(np.where(labels > 0, 255, 0))
    return labels
def _get_chars_from_back_image(back_image, chars):
    left_image = np.zeros((back_image.shape), np.uint8)
    h_sum = np.sum(back_image, axis=0)
    for i in range(len(h_sum)):
        if h_sum[i] == 255:
            back_image[:, i] = 0
    _, contours, hierarchy = cv2.findContours(back_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w > CHAR_MIN_WIDTH and w <= CHAR_MAX_WIDTH and h > CHAR_MIN_HEIGHT and h <= CHAR_MAX_HEIGHT:
            char2 = char.Char(back_image[y: y + h, x: x + w], (x, y, w, h))
            if char2.probability > 0:
                chars.append(char2)
            else:
                left_image[y: y + h, x: x + w] = back_image[y: y + h, x: x + w]
        elif w > CHAR_MAX_WIDTH or h > CHAR_MAX_HEIGHT:
            left_image[y: y + h, x: x + w] = back_image[y: y + h, x: x + w]
        else:
            continue
    return left_image
def parse_contours(binary_images):
    debug_i = 0
    if DEBUG:
        if os.path.exists(DEBUG_DIR) == False:
            os.mkdir(DEBUG_DIR)
        debug_dir = DEBUG_DIR
    shape = binary_images[0].shape
    chars = []
    for b_img in binary_images:
        b2_img = b_img.copy()
        if DEBUG:
            debug_i = write_debug(debug_dir, b2_img, debug_index=debug_i)
        b2_img = _get_chars_from_back_image(b2_img, chars)
        if np.sum(b2_img) > 0:
            b2_img = _remove_line(b2_img)
            if DEBUG:
                debug_i = write_debug(debug_dir, b2_img, debug_index=debug_i)
            b2_img = _get_chars_from_back_image(b2_img, chars)
    chars.sort(key=lambda y: y.order_key())
    # result.sort(key=lambda y: y[0])
    chars = _remove_smaller_chars(chars)
    # chars = _remove_bigger_chars(chars, shape)
    chars = _remove_chinese_chars(chars, shape)
    # last = []
    # for key, char_img, c in result:
    #     if _is_a_line(char_img):
    #         continue
    #     last.append((char_img, c))
    if DEBUG:
        for char in chars:
            debug_i = write_debug(debug_dir, char.b_img, debug_index=debug_i)
    return chars
def resize(img, width):
    # h, w = img.shape
    # radio = width / max(w, h)
    # h2, w2 = (int)(h * radio), (int)(w * radio)
    # img2 = cv2.resize(img, (w2, h2))
    # img3 = np.zeros((width, width), np.uint8)
    # if h2 < w2:
    #     img3[:h2, :] = img2
    # else:
    #     img3[:, :w2] = img2
    # return img3
    return cv2.resize(img, (width, width))
# def _hsv_query(hsv, start, end):
#     img2 = np.where(hsv <= start, 0, hsv)
#     img3 = np.where(img2 >= end, 0, 1)
#     img4 = img2 * img3
#     img5 = np.where(img4 > 0, 1, 0)
#     return np.uint8(img5)
# def get_color_image(img, color_name):
#     hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#     x, y, z = img.shape
#     result = np.ones((x, y), np.uint8)
#     if color_name == "violet":
#         result = _hsv_query(hsv[:, :, 0], 125, 155) * _hsv_query(hsv[:, :, 1], 43, 255) * _hsv_query(hsv[:, :, 2], 46, 255)
#     elif color_name == "blue":
#         result = _hsv_query(hsv[:, :, 0], 100, 124) * _hsv_query(hsv[:, :, 1], 43, 255) * _hsv_query(hsv[:, :, 2], 46, 255)
#     elif color_name == "cyan":
#         result = _hsv_query(hsv[:, :, 0], 78, 99) * _hsv_query(hsv[:, :, 1], 43, 255) * _hsv_query(hsv[:, :, 2], 46, 255)
#     elif color_name == "green":
#         result = _hsv_query(hsv[:, :, 0], 35, 77) * _hsv_query(hsv[:, :, 1], 43, 255) * _hsv_query(hsv[:, :, 2], 46, 255)
#     elif color_name == "yellow":
#         result = _hsv_query(hsv[:, :, 0], 26, 34) * _hsv_query(hsv[:, :, 1], 43, 255) * _hsv_query(hsv[:, :, 2], 46, 255)
#     elif color_name == "orange":
#         result = _hsv_query(hsv[:, :, 0], 11, 25) * _hsv_query(hsv[:, :, 1], 43, 255) * _hsv_query(hsv[:, :, 2], 46, 255)
#     elif color_name == "red":
#         result = _hsv_query(hsv[:, :, 0], 0, 10) * _hsv_query(hsv[:, :, 0], 156, 180) * _hsv_query(hsv[:, :, 1], 43, 255) * _hsv_query(hsv[:, :, 2], 46, 255)
#     elif color_name == "white":
#         result = _hsv_query(hsv[:, :, 1], 0, 30)
#     elif color_name == "black":
#         result = _hsv_query(hsv[:, :, 2], 0, 45)
#     else:   #gray
#         result = _hsv_query(hsv[:, :, 2], 46, 220)
#     return result * 255
def _is_chinese_code(char_img):
    """判断是不是汉字，如果是，则腐蚀后变成多个小对象"""
    b2_img = char_img.copy()
    w, h = b2_img.shape
    e = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    b2_img = cv2.erode(b2_img, e, iterations=1)
    _, contours, hierarchy = cv2.findContours(b2_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_r = 0
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        (x3, y3), radius = cv2.minEnclosingCircle(c)
        if radius > max_r:
            max_r = radius
    if max_r < 1.5:
        return True
    return False