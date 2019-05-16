import cv2
import numpy as np


def cv_imload(path):
    return cv2.imread(path, cv2.IMREAD_COLOR)


def cv_imshow(window_name, image, time):
    cv2.imshow(window_name, image)
    if time != -1:
        cv2.waitKey(time)


def sample_img():
    return cv2.imread("./test0.PNG", cv2.IMREAD_COLOR)


def mid_values(value_list, method):
    mid = len(value_list)
    value_list.sort(key=lambda th: method(th))
    return method(value_list[mid//2])
