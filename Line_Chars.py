# coding=utf-8
import Identifying_codes
import cv2
import utils
import char
import numpy as np
import common
import sys
class Line_Chars(Identifying_codes.Identify_code):
    def get_result(self):
        line_ids = self.get_line_ids()
        for id in line_ids:
            line_chars = []
            for char in self.get_chars():
                if char.y == id:
                    line_chars.append(char)
                    line_chars.sort(key=lambda c: c.x)
            first_char = self.get_first_char(line_chars)
            if self.is_correct(first_char):
                return self.get_codes(line_chars)
        return self.get_codes(line_chars)

    def is_correct(self, first_char):
        if first_char in self._chars:
            self._chars.remove(first_char)
        if first_char.get_color(self.get_bg_image()) == self._semantic.color:
            if self._semantic.category == "飞机" or self._semantic.category == "汽车":
                if first_char.shape() == self._semantic.shape:
                    return True
            return True
        return False
    def get_first_char(self, line_chars):
        first_char = line_chars[0]
        x, y, w, h = first_char.bbox
        if x > 10:
            start_x = 4
            x0, y0, w0, h0 = start_x, y, x - 1 - start_x, h
            shape = self._background_image.shape
            img = np.ones(shape, np.uint8) * 255
            img[y0:y0 + h0, x0:x0 + w0] = self._background_image[y0:y0 + h0, x0:x0 + w0]
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, b_img = cv2.threshold(gray, 210, 255, cv2.THRESH_BINARY_INV)
            _, contours, hierarchy = cv2.findContours(b_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            max_r = 0
            for c in contours:
                (x, y, w, h) = cv2.boundingRect(c)
                (x3, y3), radius = cv2.minEnclosingCircle(c)
                if max_r < radius:
                    first_char = char.Char(b_img[y: y + h, x: x + w], (x, y, w, h))
                    max_r = radius
        else:
            line_chars.remove(first_char)
        return first_char
# class ROI:
#     """Region of Interest"""
#     _chars = []
#     _background_image = None
#     _semantic = None
#     _roi_box = (0, 0, 0, 0)
#     def __init__(self, bg_image, roi_box, semantic):
#         self._background_image = bg_image
#         self._roi_box = roi_box
#         self._semantic = semantic
#     def __init__(self, bg_image, chars, semantic):
#         self._background_image = bg_image
#         #self._roi_box = roi_box
#         self._chars = chars
#         self._semantic = semantic
#     def resolve_chars(self):
#         if len(self._chars) == 0:
#             img = self._background_image
#             one_img = utils.bgr_to_one(img)
#             b_imgs = utils.top_k(one_img)
#             contours = utils.parse_contours(b_imgs)
#             chars = []
#             for b_img, c in contours:
#                 bbox = cv2.boundingRect(c)
#                 min_rect = cv2.minAreaRect(c)
#                 min_circle = cv2.minEnclosingCircle(c)
#                 k = char.Char(img, b_img, bbox, min_rect, min_circle)
#                 chars.append(k)
#             chars.sort(key=lambda y: y.order_key())
#             self._chars = chars
#     def get_first_char(self):
#         first_char = self._chars[0]
#         x, y, w, h = first_char.bbox
#         if x > 10:
#             start_x = 4
#             x0, y0, w0, h0 = start_x, y, x - 1 - start_x, h
#             shape = self._background_image.shape
#             img = np.ones(shape, np.uint8) * 255
#             img[y0:y0 + h0, x0:x0 + w0] = self._background_image[y0:y0 + h0, x0:x0 + w0]
#             gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#             _, b_img = cv2.threshold(gray, 210, 255, cv2.THRESH_BINARY_INV)
#             _, contours, hierarchy = cv2.findContours(b_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#             max_r = 0
#             for c in contours:
#                 (x, y, w, h) = cv2.boundingRect(c)
#                 min_box = cv2.minAreaRect(c)
#                 (x3, y3), radius = cv2.minEnclosingCircle(c)
#                 if max_r < radius:
#                     first_char = char.Char(self._background_image, b_img[y: y + h, x: x + w], (x, y, w, h), min_box, ((x3, y3), radius))
#                     max_r = radius
#         return first_char
#     def is_correct(self):
#         first_char = self.get_first_char()
#         if first_char in self._chars:
#             self._chars.remove(first_char)
#         if first_char.color() == self._semantic.color:
#             if self._semantic.category == "飞机" or self._semantic.category == "汽车":
#                 if first_char.shape() == self._semantic.shape:
#                     return True
#             return True
#         return False
#     def get_result(self):
#         self._chars.sort(key=lambda y: y.order_value(self._semantic.direction))
#         common.crack_chars(self._chars)
#         result = ""
#         for c in self._chars:
#             if c.prob > 0:
#                 result = result + c.code
#         start = self._semantic.start_index
#         end = self._semantic.end_index
#         if end < 0:
#             return result[end:]
#         else:
#             if end > start:
#                 return result[start:end]
#             return result
#     def get_color(self,img):
#         result = "unknown_color"
#         colors = utils.init_colors()
#         B, G, R = img[:, :, 0], img[:, :, 1], img[:, :, 2]
#         one_img = (np.uint32(R) << 16) + (np.uint32(G) << 8) + B
#         x, y = one_img.shape
#         h = {}
#         for i in range(x):
#             for j in range(y):
#                 p = one_img[i, j]
#                 if p != 16777215:  # except white color 255,255,255
#                     if p in h.keys():
#                         h[p] = h[p] + 1
#                     else:
#                         h[p] = 0
#         pixels = []
#         for k, v in h.items():
#             pixels.append((k, v))
#         pixels.sort(key=lambda y: y[len(y) - 1])
#
#         (color_value, color_count) = pixels[0]
#         color_value = np.uint32(color_value)
#         b = np.uint32(color_value << 24) >> 24
#         g = np.uint32(color_value << 16) >> 24
#         r = np.uint32(color_value << 8) >> 24
#         distance = 99999999
#         for c in colors:
#             for (r2, g2, b2) in c.rgbs:
#                 r3 = (r - r2) * (r - r2)
#                 g3 = (g - g2) * (g - g2)
#                 b3 = (b - b2) * (b - b2)
#                 distance2 = r3 + g3 + b3
#                 if distance2 < distance:
#                     result = c.name
#                     distance = distance2
#         return result
if __name__ == '__main__':
    file_name = sys.argv[1]
    semantic = sys.argv[2]
    img = cv2.imread(file_name)
    family = Line_Chars(img, semantic)
    print(family.get_result())