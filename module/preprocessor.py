import numpy as np
import cv2
from module.util import *
from module.data import *


def make_rect(points):
    lx = points[0][0][0]
    ly = points[0][0][1]
    rx = points[0][0][0]
    ry = points[0][0][1]
    for pt in points:
        pt = pt[0]
        if lx > pt[0]:
            lx = pt[0]
        if rx < pt[0]:
            rx = pt[0]

        if ly > pt[1]:
            ly = pt[1]

        if ry < pt[1]:
            ry = pt[1]
    return Rectangle([(lx, ly), (rx, ry)])


def draw_roi(image, roi, color, thin):
    return cv2.rectangle(image, roi[0], roi[1], color, thin)


def points_to_rectangles(contours):
    rectangles = []
    for points in contours:
        rect = make_rect(points)
        rectangles.append(rect)
    return rectangles


def img_to_binary(image):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return image


def size_filter(region, arg):
    return region.area > arg[0]


# find object roi
def make_regions(image, filters, filter_args, reverse=True):
    image = img_to_binary(image)
    if reverse:
        image = 255-image
    _, image = cv2.threshold(image, 125, 255, cv2.THRESH_BINARY)
    _, contours, _ = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    basic_roi = points_to_rectangles(contours)
    final_roi = []
    for roi in basic_roi:
        filter_ok = True
        for idx, roi_filter in enumerate(filters):
            filter_ok = roi_filter(roi, filter_args[idx])
            if not filter_ok:
                break
        if filter_ok:
            final_roi.append(roi)
    final_roi.sort(key=lambda roi: roi.area)
    return final_roi


def union_intersection(img, regions):
    bin_img = img_to_binary(img)

    for region in regions:
        bin_img = draw_roi(bin_img, region.points, 0, -1)
        bin_img = draw_roi(bin_img, region.points, 0, 2)

    regions = make_regions(bin_img, [size_filter], [[10, ]])
    regions.sort(key=lambda roi: roi.area)
    return regions


def split_horizontal_regions(img, regions):
    bin_img = img_to_binary(img)
    height, width = bin_img.shape

    board = np.zeros((height, width), dtype="uint8")
    board = 255 + board

    for region in regions:
        new_points = [(0, region.points[0][1]), (width, region.points[1][1])]
        board = draw_roi(board, new_points, 0, -1)

    region = make_regions(board, [], [])

    return region


def erase_region(img, regions):
    color = 255
    if len(img.shape) == 3:
        color = [255, 255, 255]

    erase_regions = regions[-3:]

    for region in erase_regions:
        img = draw_roi(img, region.points, color, -1)
    return img, regions[:-3]


if __name__ == "__main__":
    input_img = sample_img()
    roi_img = input_img.copy()
    rois = make_regions(input_img, [size_filter], [[10, ]])

    for roi in rois:
        roi_img = draw_roi(roi_img, roi.points, [0, 255, 0], 1)

    cv_imshow("roi", roi_img, 0)

    rois = union_intersection(input_img, rois)

    roi_img = input_img.copy()
    cv_imshow("roi", roi_img, 0)

    for roi in rois:
        roi_img = draw_roi(roi_img, roi.points, [0, 255, 0], 1)

    cv_imshow("roi", roi_img, 0)

    erased_img, rois = erase_region(input_img, rois)

    cv_imshow("erase", erased_img, 0)
    rois = split_horizontal_regions(erased_img, rois)

    roi_img = input_img.copy()
    for roi in rois:
        roi_img = draw_roi(roi_img, roi.points, [0, 255, 0], 1)

    cv_imshow("roi", roi_img, 0)

