# coding=utf-8
import cv2
import utils
import common
Y_ORDER_ERROR_DELTA = 20
X_ORDER_ERROR_DELTA = 3
IMAGE_RECT = 16
class Char:
    def __init__(self, binary_img, bbox):
        self.b_img = binary_img
        self.bbox = bbox
        self.x, self.y = self.get_xy()
        self.color = "unknown"
        self.code, self.probability = self.get_code()
        self.shape = self.get_shape()
        self.is_first = self._is_first()
        self._image = None
    def order_key(self, direction=[]):
        x, y, w, h = self.bbox
        if len(direction) > 0:
            if direction[0] == "上" or direction[0] == "高":
                return y
            if direction[0] == "下" or direction[0] == "低":
                return 0 - y
            if direction[0] == "左" or direction[0] == "前":
                return x
            if direction[0] == "右" or direction[0] == "后":
                return 0 - x
        return ((self.y + 1) << 16) + self.x + 1
    def get_shape(self):
        x, y, w, h = self.bbox
        if utils.approximate(w, h, delta=0.15):
            return "正方"
        else:
            return "长方"
    def get_xy(self):
        (x, y, w, h) = self.bbox
        return int(x / X_ORDER_ERROR_DELTA + 0.5), int(y / Y_ORDER_ERROR_DELTA + 0.5)

    def get_color(self, back_image):
        (x, y, w, h) = self.bbox
        bgr_img = back_image[y: y + h, x: x + w]
        return utils.analyze_image_color(bgr_img)
    def get_code(self):
        char2 = self.get_image()
        return common.crack_char(char2)
    def confirmed_code(self):
        char2 = self.get_image()
        if self.probability < 0:
            return ""
        for i in range(2):
            code2, prob2 = common.crack_char(char2)
            if code2 != self.code or prob2 < 0:
                return ""
        return self.code
    def get_image(self):
        self._image = cv2.resize(self.b_img, (IMAGE_RECT, IMAGE_RECT))
        return self._image
    def _is_first(self):
        (x, y, w, h) = self.bbox
        if x < 10:
            return True
        return False





