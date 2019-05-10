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


def draw_image_region(image, image_region, color, thin):
    return cv2.rectangle(image, image_region[0], image_region[1], color, thin)


def points_to_rectangles(contours):
    rectangles = []
    for points in contours:
        rect = make_rect(points)
        rectangles.append(rect)
    return rectangles


def img_to_binary(image):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def size_filter(region, arg):
    return region.area > arg[0]


# find object image_region
def make_regions(image, filters, filter_args, reverse=True):
    image = img_to_binary(image)
    if reverse:
        image = 255-image
    _, image = cv2.threshold(image, 5, 255, cv2.THRESH_BINARY)
    _, contours, _ = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    basic_image_region = points_to_rectangles(contours)
    final_image_region = []
    filter_image_region = []
    for image_region in basic_image_region:
        filter_ok = True
        for idx, image_region_filter in enumerate(filters):
            filter_ok = image_region_filter(image_region, filter_args[idx])
            if not filter_ok:
                filter_image_region.append(image_region)
                break
        if filter_ok:
            final_image_region.append(image_region)
    final_image_region.sort(key=lambda image_region: image_region.area)
    return final_image_region, filter_image_region


def union_intersection(img, regions):
    bin_img = img_to_binary(img)

    for region in regions:
        bin_img = draw_image_region(bin_img, region.points, 0, -1)
        bin_img = draw_image_region(bin_img, region.points, 0, 2)

    regions, _ = make_regions(bin_img, [size_filter], [[10, ]])
    regions.sort(key=lambda image_region: image_region.area)
    return regions


def split_horizontal_regions(img, regions):
    bin_img = img_to_binary(img)
    height, width = bin_img.shape

    board = np.zeros((height, width), dtype="uint8")
    board = 255 + board

    for region in regions:
        new_points = [(0, region.points[0][1]), (width, region.points[1][1])]
        board = draw_image_region(board, new_points, 0, -1)

    horizontal_region, _ = make_regions(board, [], [])
    horizontal_region.sort(key=lambda image_region: image_region.points[0][1])
    return horizontal_region


def erase_region(img, regions):
    color = 255
    if len(img.shape) == 3:
        color = [255, 255, 255]
    mid = mid_values(regions, lambda object: object.area)
    print("mid {}".format(mid))
    erase_regions = []
    new_regions = []
    for image_region in regions:
        if image_region.area > mid*10:
            erase_regions.append(image_region)
        else:
            new_regions.append(image_region)

    for region in erase_regions:
        img = draw_image_region(img, region.points, color, -1)

    return img, new_regions


def union_region(r1, r2):
    return Rectangle([(r1.points[0]), (r2.points[1])])


def split_vertical_regions(img):
    bin_img = img_to_binary(img)
    height, width = bin_img.shape

    image_region_img = bin_img.copy()
    image_region_img = cv2.resize(image_region_img, (width//height * 30, 30), cv2.INTER_CUBIC)
    word_image = image_region_img.copy()

    basic_regions, _ = make_regions(image_region_img, [], [])

    union_region = union_intersection(image_region_img, basic_regions)
    union_region.sort(key=lambda region:region.points[0][0])
    word_images = []
    for region in union_region:
        pts = [(region.points[0][0], 0), (region.points[1][0], 30)]
        image_region_img = draw_image_region(image_region_img, pts, 0, -1)

    vertical_regions, _ = make_regions(image_region_img, [], [])
    vertical_regions.sort(key=lambda region:region.points[0][0])
    word_regions = []
    for region in vertical_regions:
        pts = [(region.points[0][0], 0), (region.points[1][0], 30)]
        temp = word_image[pts[0][1]:pts[1][1], pts[0][0]:pts[1][0]]

        temp_region, _ = make_regions(temp, [], [])
        temp_points = []

        for roi in temp_region:
            for pt in roi.points:
                temp_points.append([pt])
        final_roi = make_rect(temp_points)
        final_roi.points[0] = (final_roi.points[0][0], final_roi.points[0][1])
        final_roi.points[1] = (final_roi.points[1][0]+2, final_roi.points[1][1]+2)

        pts = final_roi.points
        if final_roi.ratio > 1.5:
            pts = final_roi.points
            diff = (pts[1][0]-pts[0][0])//2
            pts1 = [(pts[0][0], pts[0][1]), (pts[0][0]+diff, pts[1][1])]
            pts2 = [(pts[0][0]+diff, pts[0][1]), (pts[1][0], pts[1][1])]
            word_regions.append(Rectangle(pts1))
            word_regions.append(Rectangle(pts2))
            word1 = temp[pts1[0][1]:pts1[1][1], pts1[0][0]:pts1[1][0]]
            word2 = temp[pts2[0][1]:pts2[1][1], pts2[0][0]:pts2[1][0]]
            word_images.append(word1)
            word_images.append(word2)

        else:
            pts = final_roi.points
            word_regions.append(final_roi)
            word_images.append(temp[pts[0][1]:pts[1][1], pts[0][0]:pts[1][0]])

    return word_images, word_regions


def image_regular(image):
    binary_image = img_to_binary(image)
    roi = cv2.resize(binary_image, (54, 54), cv2.INTER_CUBIC)
    board = np.zeros((64, 64), dtype="uint8")
    board = 255 + board
    board[5:59, 5:59] = roi
    cv2.destroyWindow("board")
    _, board = cv2.threshold(board,125, 255, cv2.THRESH_OTSU)
    cv_imshow("board", board, 0)
    return board


if __name__ == "__main__":
    input_img = sample_img()
    image_region_img = input_img.copy()
    image_regions, _ = make_regions(input_img, [size_filter], [[10, ]])
    image_regions = union_intersection(input_img, image_regions)
    erased_img, image_regions = erase_region(input_img, image_regions)
    vertical_regions = split_horizontal_regions(erased_img, image_regions)

    word_images = []
    word_regions = []

    for image_region in vertical_regions:
        points = image_region.points
        words, temp_region = split_vertical_regions(erased_img[points[0][1]:points[1][1], points[0][0]:points[1][0]])
        word_images = word_images + words
        word_regions = word_regions + temp_region

    for idx, word in enumerate(word_images):
        image_regular(word)
        print(word_regions[idx].ratio)

