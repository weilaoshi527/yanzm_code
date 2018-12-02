# coding=utf-8

import Identifying_codes
import utils
class Frame_Chars(Identifying_codes.Identify_code):
    def get_result(self):
        rects = self.get_rects()
        for rect in rects:
            x, y, w, h = rect
            img = self.get_bg_image()[y:y + 2, x:x + 2, :]
            color_name = utils.analyze_image_color(img)
            if color_name == self._semantic.color:
                rect_chars = []
                for char in self.get_chars():
                    if self.get_semantic().in_out == "内":
                        if utils.contains(rect, char.bbox):
                            rect_chars.append(char)
                    else:
                        if utils.contains(rect, char.bbox) == False:
                            rect_chars.append(char)
                return self.get_codes(rect_chars)
        return self.get_codes(self.get_chars())
    def get_rects(self):
        facked_rects = self.get_contour_rects()
        chars = []
        rects = []
        for r in facked_rects:
            x, y, w, h = r
            if len(chars) == 0:
                chars.append(r)
            else:
                ave_r = 0
                for r2 in chars:
                    x2, y2, w2, h2 = r2
                    ave_r = ave_r + max(h2, w2)
                ave_r = ave_r / len(chars)
                if utils.approximate(h, ave_r):
                    chars.append(r)
                else:
                    rects.append(r)
        return rects
