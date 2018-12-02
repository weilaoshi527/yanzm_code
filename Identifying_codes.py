# coding=utf-8
import cv2
import numpy as np
import utils
import semantic_parser as sp

class Identify_code:
    _background_image = None
    _semantic = ""
    _chars = []
    def __init__(self, bg_image, semantic):
        self._background_image = utils._remove_bounding_box(bg_image)
        self._semantic = sp.Semantic_parser(semantic)
        self._chars = self.get_chars()
    def get_binary_image(self, image = _background_image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        return binary
    def get_chars(self):
        if self._chars == []:
            img = self._background_image
            one_img = utils.bgr_to_one(img)
            b_imgs = utils.top_k(one_img)
            chars = utils.parse_contours(b_imgs)
            self._chars = chars
            self.remove_bad_chars()
        return self._chars
    def remove_bad_chars(self):
        line_ids = self.get_line_ids()
        bad_ids = []
        for id in line_ids:
            count = 0
            for char in self.get_chars():
                if char.y == id:
                    count = count + 1
            if count == 1:              #如果找不到小伙伴，可能是孤立的错误字符
                bad_ids.append(id)
        for bad_id in bad_ids:
            for char in self.get_chars():
                if char.y == bad_id:
                    self.remove_bad_char(char)
    def remove_bad_char(self, char):
        self._chars.remove(char)
    def get_contour_rects(self):
        binary = self.get_binary_image(self._background_image)
        _, contours, hierarchy = cv2.findContours(binary.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rects = []
        for c in contours:
            rect = cv2.boundingRect(c)
            rects.append(rect)
        return rects
    def get_line_ids(self):
        y_ids = []
        for char in self._chars:
            if char.y not in y_ids:
                y_ids.append(char.y)
        return y_ids
    def get_line_count(self):
        return len(self.get_line_ids())
    def get_debug_image(self):
        x, y, z = self._background_image.shape
        debug_image = np.zeros((x, y), np.uint8)
        for c in self._chars:
            x, y, w, h = c.bbox
            debug_image[y: y + h, x: x + w] = c.b_img
        return debug_image
    def get_result(self):
        return self.get_codes(self._chars)
    def get_codes(self, line_chars):
        line_chars.sort(key=lambda char: char.order_key(self._semantic.direction))
        result = ""
        for c in line_chars:
            result = result + c.code
        start = self._semantic.start_index
        end = self._semantic.end_index
        if end < 0:
            return result[end:]
        else:
            if end > start:
                return result[start:end]
            return result
    def get_bg_image(self):
        return self._background_image
    def get_semantic(self):
        return self._semantic

