import cv2
import os
import pytesseract as tess
import numpy as np

def recognize_text(img):
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, binnary = cv2.threshold(gray, 100, 255, cv2.THRESH_OTSU)
    kerhel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    bin1 = cv2.morphologyEx(binnary, cv2.MORPH_OPEN, kerhel1, iterations=1)
    kerhel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
    bin2 = cv2.morphologyEx(binnary, cv2.MORPH_OPEN, kerhel2, iterations=1)
    text = tess.image_to_string(bin2)
    """
    text = tess.image_to_string(img)
    print("识别结果: {0}".format(text))

def knn_text(img):
    data = np.loadtxt('letter-recognition.data', dtype= 'float32', delimiter = ',',
          converters= {0: lambda ch: ord(ch)-ord('A')})
    # split the data to two, 10000 each for train and test
    train, test = np.vsplit(data,2)
    # split trainData and testData to features and responses
    responses, trainData = np.hsplit(train,[1])
    labels, testData = np.hsplit(test,[1])
    # Initiate the kNN, classify, measure accuracy.
    knn = cv2.ml.KNearest_create()
    knn.train(trainData, cv2.ml.ROW_SAMPLE, responses)
    ret, result, neighbours, dist = knn.findNearest(testData, k=5)
    correct = np.count_nonzero(result == labels)
    accuracy = correct*100.0/10000
    print(accuracy)

def get_img_h_w(img):
    return (img.shape[0], img.shape[1])

def get_roi(path):
    img = cv2.imread(path)
    (height, width) = get_img_h_w(img)
    roi_height = height - 64
    roi_width = width
    roi = img[roi_height:height, 0:roi_width]
    return roi

def get_limit_speed(img):
    (height, width) = get_img_h_w(img)
    roi_height_start = 0
    roi_height_end = 30
    roi_width_start = width - 500
    roi_width_end = width - 450
    roi = img[roi_height_start:roi_height_end, roi_width_start:roi_width_end]
    return roi

def get_speed(img):
    (height, width) = get_img_h_w(img)
    roi_height_start = 0
    roi_height_end = 30
    roi_width_start = width - 180
    roi_width_end = width - 130
    roi = img[roi_height_start:roi_height_end, roi_width_start:roi_width_end]
    return roi

def get_distance(img):
    (height, width) = get_img_h_w(img)
    roi_height_start = 0
    roi_height_end = 30
    roi_width_start = width - 1796
    roi_width_end = width - 1665
    roi = img[roi_height_start:roi_height_end, roi_width_start:roi_width_end]
    return roi

def get_enter_time(img):
    (height, width) = get_img_h_w(img)
    roi_height_start = 0
    roi_height_end = 30
    roi_width_start = width - 1074
    roi_width_end = width - 700
    roi = img[roi_height_start:roi_height_end, roi_width_start:roi_width_end]
    return roi

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

def name_plus_index(path, index):
    name, ext = os.path.splitext(path)
    return name + "_" + str(index) + ext

def get_test(get_f):
    src_root = "./test"
    dst_root = "./dst"
    for file_name in os.listdir(src_root):
        src = os.path.join(src_root, file_name)
        dst = os.path.join(dst_root, file_name)
        roi = get_roi(src)
        roi = get_f(roi)
        if roi.size > 0:
            cv2.imwrite(dst, roi)
            n_rois = split_number(roi)
            for i, n_roi in enumerate(n_rois):
                if n_roi.size == 0:
                    continue
                cv2.imwrite(name_plus_index(dst, i), n_roi)
            """
            recognize_text(roi)
            """
            """
            n_rois = split_number(roi)
            # for n_roi in n_rois:
                # recognize_text(n_roi)
            for i, n_roi in enumerate(n_rois):
                if n_roi.size == 0:
                    continue
                # recognize_text(n_roi)
                cv2.imwrite(name_plus_index(dst, i), n_roi)
            # cv2.imwrite(dst, roi)
            """

def get_limit_speed_test():
    get_test(get_limit_speed)

def get_speed_test():
    get_test(get_speed)

def get_distance_test():
    get_test(get_distance)

def get_enter_time_test():
    get_test(get_enter_time)

def recognize_test():
    samples = np.loadtxt('../knn/generalsamples.data',np.float32)
    responses = np.loadtxt('../knn/generalresponses.data',np.float32)
    responses = responses.reshape((responses.size,1))
    model = cv2.ml.KNearest_create()
    model.train(samples,cv2.ml.ROW_SAMPLE,responses)

# recognize_test()

# get_limit_speed_test()
# get_speed_test()
get_distance_test()
# get_enter_time_test()

"""
roi = get_roi("./test.jpg")
roi = get_speed(roi)
cv2.imwrite("./roi.png", roi)
"""

# cv2.imwrite("./out.png", img)
# print("hight: {0}, width: {1}".format(height, width))

# cv2.imshow("test", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
