# coding=utf-8
import re

class Semantic_parser:
    is_line = False
    is_rect = False
    start_index = 0
    end_index = 0
    category = ""
    shape = ""
    color = ""
    direction = []
    in_out = "内"
    _semantic = ""
    def __init__(self, semantic):
        self._semantic = semantic
        self.parse_semantic()
    def parse_semantic(self):
        semantic = self._semantic
        number_str = "(1|2|3|4|5|6|7|89|一|二|两|三|四|五|六|七|八|九|壹|贰|叁|肆|伍|陆|柒|捌|玖)+"
        category_str = "(苹果|香蕉|飞机|西瓜|汽车|绿灯|红灯|\w色\w*)+"
        color_str = "(红|橙|黄|绿|青|蓝|紫|黑)+"
        shape_str = "(圆|长方|正方|矩|三角|菱)+"
        direct_str = "(上|下|左|右|前|后|高|低)+"
        in_out_str = "(内|外)"
        category2Color = {}
        category2Color["苹果"] = "红"
        category2Color["香蕉"] = "黄"
        category2Color["飞机"] = "黄"
        category2Color["西瓜"] = "绿"
        category2Color["汽车"] = "黄"
        category2Color["绿灯"] = "绿"
        category2Color["红灯"] = "红"
        category2Shape = {}
        category2Shape["苹果"] = "正方"
        category2Shape["香蕉"] = "正方"
        category2Shape["飞机"] = "正方"
        category2Shape["西瓜"] = "正方"
        category2Shape["汽车"] = "长方"
        category2Shape["绿灯"] = "正方"
        category2Shape["红灯"] = "正方"
        from_to_pattern = re.compile('第' + number_str + '到第' + number_str)
        front_pattern = re.compile('前' + number_str)
        back_pattern = re.compile('后' + number_str)
        category_pattern = re.compile(category_str + "行")
        color_pattern = re.compile(color_str + "色")
        shape_pattern = re.compile(shape_str + "形")
        # color_shape_pattern = re.compile(color_str + "色" + shape_str + "形" + "行")
        direct_pattern = re.compile("\w{1}" + direct_str + "\w{1}" + direct_str)
        rect_pattern = re.compile("框" + in_out_str)
        result = re.findall(from_to_pattern, semantic)
        if len(result) == 1:
            self.start_index, self.end_index = self._tranlate_number(result[0][0], result[0][1])
        result = re.findall(front_pattern, semantic)
        if len(result) == 1:
            self.start_index, self.end_index = self._tranlate_number(1, result[0])
        result = re.findall(back_pattern, semantic)
        if len(result) == 1:
            self.start_index, self.end_index = self._tranlate_number(-1, result[0])
        result = re.findall(category_pattern, semantic)
        if len(result) == 1:
            self.category = result[0]
            if self.category in category2Color.keys():
                self.color = category2Color[self.category]
            if self.category in category2Shape.keys():
                self.shape = category2Shape[self.category]
            self.is_line = True
        result = re.findall(color_pattern, semantic)
        if len(result) == 1:
            self.color = result[0]
        result = re.findall(shape_pattern, semantic)
        if len(result) == 1:
            self.shape = result[0]
        result = re.findall(direct_pattern, semantic)
        if len(result) == 1:
            self.direction.clear()
            self.direction.append(result[0][0])
            self.direction.append(result[0][1])
        result = re.findall(rect_pattern, semantic)
        if len(result) == 1:
            self.in_out = result[0]
            self.is_rect = True
    def _tranlate_number(self, start_num, end_num):
        number_dict = {"一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, }
        number_dict["壹"] = "1"
        number_dict["贰"] = "2"
        number_dict["叁"] = "3"
        number_dict["肆"] = "4"
        number_dict["伍"] = "5"
        number_dict["陆"] = "6"
        number_dict["柒"] = "7"
        number_dict["捌"] = "8"
        number_dict["玖"] = "9"
        if start_num in number_dict.keys():
            start = ((int)(number_dict[start_num]) - 1)
        else:
            start = ((int)(start_num) - 1)
        if end_num in number_dict.keys():
            end = ((int)(number_dict[end_num]))
        else:
            end = ((int)(end_num))
        if start < 0:
            end = 0 - end
        return start, end