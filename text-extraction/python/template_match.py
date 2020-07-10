import cv2
import numpy as np
import os

def get_img_h_w(img):
    return (img.shape[0], img.shape[1])

def split_number(img):
    (height, width) = get_img_h_w(img)
    n_len = 16
    times = int(width / n_len)
    rois = []
    for i in range(times):
        roi_height_start = 0 
        roi_height_end = height
        roi_width_start = i * n_len + 2 
        roi_width_end = roi_width_start + n_len
        roi = img[roi_height_start:roi_height_end, roi_width_start:roi_width_end]
        rois.append(roi)
    return rois

def load_template(template_root):
    r = []
    for file_name in os.listdir(template_root):
        img = cv2.imread(os.path.join(template_root, file_name), cv2.IMREAD_GRAYSCALE)
        ret, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        name, _ = os.path.splitext(file_name)
        r.append((img, name))
    return r

def img_pretreatment(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    return img

def template_match_path_input(src_path, template_path):
    template_img = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
    src_img = cv2.imread(src_path, cv2.IMREAD_GRAYSCALE)
    # src_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)
    return template_match(src_img, template_img)

def template_match(src_img, template_img):
    template_ret, template_img = cv2.threshold(template_img, 127, 255, cv2.THRESH_BINARY)
    src_ret, src_img = cv2.threshold(src_img, 127, 255, cv2.THRESH_BINARY)
    # cv2.imwrite("./src_threshold.jpg", src_img)
    # cv2.imwrite("./template_threshold.jpg", template_img)
    # 标准平方差匹配
    # method = cv2.TM_SQDIFF_NORMED
    # 标准相关匹配
    method = cv2.TM_CCORR_NORMED
    # 标准相关系数匹配
    # method = cv.TM_CCOEFF_NORMED
    template_height, template_width = template_img.shape[:2]
    result = cv2.matchTemplate(src_img, template_img, method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    if method == cv2.TM_SQDIFF_NORMED:
        loc = min_loc
        val = min_val
    else:
        loc = max_loc
        val = max_val
    """
    br = (tl[0] + template_width, tl[1] + template_height);
    target_img = src_img.copy()
    cv2.rectangle(target_img, tl, br, (255, 0, 0), 2)
    cv2.imwrite("./out.png", target_img)
    """
    return (val, loc)

# 一张原图, 使用模板图片依次与原图比较, 并记录匹配的位置
# 缺点: 如果数值很相似, 将识别不准确
def get_value_v1(src_path, template_root):
    unorder = {}
    for file_name in os.listdir(template_root):
        name, _ = os.path.splitext(file_name)
        pos = template_match_path_input(src_path, os.path.join(template_root, file_name))
        if pos is None:
            continue
        width = pos[0]
        unorder[width] = name
    order = sorted(unorder.items(), key = lambda unorder:unorder[0], reverse = False)
    print(order)
    """
    if isinstance(name, int):
        number = int(name)
    if isinstance(name, str):
        if name == "point":
    print(name)
    """
    # sorted(mydict.items(),key = lambda mydict:mydict[1],reverse= False)

# 将原图分割成一个个的数字, 然后每一个数字都和所有的模板匹配, 对所有的匹配结果进行排序, 得到最大值
def get_value_v2(src_path, templates):
    img = cv2.imread(src_path)
    rois = split_number(img)
    result = []
    for roi in rois:
        roi = img_pretreatment(roi)
        # 依次比对
        unorder = []
        for (t_img, t_value) in templates:
            (val, _) = template_match(roi, t_img)
            unorder.append((val, t_value))
        order = sorted(unorder, key = lambda item:item[0], reverse = True)
        if len(order) == 0:
            continue
        _, max_value = order[0]
        if max_value == "point":
            max_value = "."
        if max_value == "colon":
            max_value = ":"
        if max_value == "mid-line":
            max_value = "-"
        result.append(max_value)
    result = "".join(result)
    return result

def get_value_v1_test():
    get_value_v1("template_test/test1.jpg", "template")

def get_value_v2_test():
    ts = load_template("template")
    r = get_value_v2("template_test/test2.jpg", ts)
    print(r)

# get_value_v1_test()
get_value_v2_test()

# template_match_path_input("template_test/test1.jpg", "template/8.jpg")
