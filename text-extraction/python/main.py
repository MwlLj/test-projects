import os
import template_match
import finder

from enum import Enum

class Value(Enum):
    limit_speed = 1
    speed = 2

def get_item(img, get_f, ts):
    roi = get_f(img)
    if roi.size == 0:
        return None
    return template_match.get_value_v2(roi, ts)

def get():
    src_root = "./test"
    dst_root = "./dst"
    template_root = "./template"
    ts = template_match.load_template(template_root)
    values = []
    for file_name in os.listdir(src_root):
        src = os.path.join(src_root, file_name)
        dst = os.path.join(dst_root, file_name)
        roi = finder.get_roi(src)
        vs = {}
        # limit speed
        value = get_item(roi, finder.get_limit_speed, ts)
        vs[Value.limit_speed] = value
        # speed
        value = get_item(roi, finder.get_speed, ts)
        vs[Value.speed] = value
        values.append(vs)
    return values

def main():
    values = get()
    for vs in values:
        print(vs)

if __name__ == '__main__':
    main()
