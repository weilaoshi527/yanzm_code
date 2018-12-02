# coding=utf-8
import Identifying_codes
import cv2
import sys
class Simple_Chars(Identifying_codes.Identify_code):
     _author = "zuohuaiyu"
#     def get_ROIs(self):
#         img = self.get_bg_image()
#         x, y, z = img.shape
#         roi = ROI(img, (0, 0, x, y),self.get_semantic())
#         self.add_roi(roi)
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
#     def resolve_chars(self):
#         img = self._background_image
#         one_img = utils.bgr_to_one(img)
#         b_imgs = utils.top_k(one_img)
#         chars = utils.parse_contours(b_imgs)
#         self._chars = chars
#     def is_correct(self):
#         return True
#     def get_result(self):
#         result = ""
#         for c in self._chars:
#             result = result + c.code
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
    family = Simple_Chars(cv2.imread(file_name), semantic)
    print(family.get_result())