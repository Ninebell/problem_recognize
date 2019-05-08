import cv2


def cv_imshow(window_name, image, time):
    cv2.imshow(window_name, image)
    if time != -1:
        cv2.waitKey(time)


def sample_img():
    return cv2.imread("../test0.PNG", cv2.IMREAD_COLOR)
