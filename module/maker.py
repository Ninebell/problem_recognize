#encoding=utf-8

import shutil
from PIL import ImageFont, ImageDraw, Image
from os import listdir, makedirs
from os.path import isfile, join
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates
import cv2
from module.preprocessor import image_regular
import numpy as np
import random
from module import preprocessor

DATA_FOLDER_PATH = "C:/Users/Jonghoe/PycharmProjects/problem_recognize/data"
LABEL_PATH = DATA_FOLDER_PATH+"/labels/useful_hangul_label.txt"


def useful_label():
    f = open(LABEL_PATH, 'r')
    file_info = f.read()
    lines = file_info.split("\n")
    return lines


def get_new_fonts():
    font_path = DATA_FOLDER_PATH+"/new_fonts"
    fonts = [f for f in listdir(font_path) if isfile(join(font_path, f))]
    for i, font in enumerate(fonts):
        fonts[i] = font_path+"/"+font
        print(fonts[i])

    return fonts


def get_fonts():
    font_path = DATA_FOLDER_PATH+"/fonts"
    fonts = [f for f in listdir(font_path) if isfile(join(font_path, f))]
    for i, font in enumerate(fonts):
        fonts[i] = font_path+"/"+font
        print(fonts[i])
    return fonts


def hangulFilePathImageRead(filePath):
    stream = open(filePath.encode("utf-8"), "rb")
    bytes = bytearray(stream.read())
    numpyArray = np.asarray(bytes, dtype=np.uint8)
    return cv2.imdecode(numpyArray, cv2.IMREAD_UNCHANGED)


def make_dir(dir_path):
    labels = useful_label()
    exist_dir = listdir(dir_path)
    for label in labels:
        if label not in exist_dir:
            dir_name = dir_path+"/"+label
            dir_name = dir_name.encode()
            print("make folder {}".format(dir_name))
            makedirs(dir_name)


def divide_font(src_path, dst_path, font_count):
    labels = useful_label()
    src_labels = listdir(src_path)
    src_labels.sort(key=lambda body: int(body.split('_')[1].split('.')[0]))

    for i, image in enumerate(src_labels):
        label_idx = i // font_count
        print(dst_path+"\\"+labels[label_idx]+"\\"+image)
        img = cv2.imread(src_path+"\\"+image, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
        img = preprocessor.image_regular(img)
        path = dst_path+"\\"+labels[label_idx]+"\\"
        folder = listdir(path)
        count = len(folder) + 1
        count = "{}".format(count)
        count = count.zfill(4)
        save_path = "{}{}{}.png".format(path, labels[label_idx], count)
        save_path = save_path.encode()
        save_img = Image.fromarray(img)
        save_img.save(save_path, "PNG")
        print(save_path)


def make_dataset():

    fonts = get_new_fonts()
    if len(fonts) == 0:
        print("new font doesn't exist.")
        return None

    labels = useful_label()
    name_dict = {}
    for i, label in enumerate(labels):
        label = label.replace("_", "")
        name_dict[i] = label

    for name in name_dict:
        print(name_dict[name])

    dataset_path = DATA_FOLDER_PATH+"/dataset/font_label"
    han_word_paths = listdir(dataset_path)

    board = np.zeros((64, 64, 3), dtype="uint8")
    board = 255 + board
    all_data_count = 0
    stop = False

    print("new font 수: {}".format(len(fonts)))

    fonts.sort()
    print(fonts)
    for i, label in name_dict.items():
        font_count = 0
        dir_path = ""
        for path in han_word_paths:
            if label == path:
                dir_path = path
                break
            if label == path+"_":
                dir_path = path
                break
        print("{} {}".format(label, dir_path))
        for font_path in fonts:
            font_size = 45
            loc_x = 5
            loc_y = 5

            font = ImageFont.truetype(font_path, font_size)
            img_pil = Image.fromarray(board)

            draw = ImageDraw.Draw(img_pil)
            draw.text((loc_x, loc_y), label, font=font, fill=(0, 0, 0, 0))
            img = np.array(img_pil)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, img = cv2.threshold(img, 125, 255, cv2.THRESH_BINARY)

            font_name = font_path.split('/')[-1]
            reg_img = image_regular(img)

            save_path = dataset_path+"/"+dir_path
            count = len(listdir(save_path))+1
            count_str = "{:04d}".format(count)

            #count_str = count_str.zfill(4)

            save_path = "{}/{}{}.png".format(save_path, label, count_str)

            save_path = save_path.encode()
            save_img = Image.fromarray(reg_img)
            save_img.save(save_path, "PNG")
            print("{} || {}, {}: {}".format(save_path, font_name, count_str, label))
            img = 255 - img
            for j in range(0, 3):
                distorted_array = elastic_distort(
                    img, alpha=random.randint(30, 36),
                    sigma=random.randint(5, 6)
                )

                distorted_array = 255 - distorted_array
                reg_img = image_regular(distorted_array)
                save_path = dataset_path+"/"+dir_path

                count = len(listdir(save_path))+1
                count_str = "{}".format(count)
                count_str = count_str.zfill(4)

                save_path = "{}/{}{}.png".format(save_path, label, count_str)
                save_path = save_path.encode()

                save_img = Image.fromarray(reg_img)
                save_img.save(save_path, "PNG")

            font_count = count
        all_data_count = all_data_count + font_count

    print("all data count: {}".format(all_data_count))
    for font in fonts:
        font_name = font.split('/')[-1]
        shutil.move(font, DATA_FOLDER_PATH+"/fonts/"+font_name)


def elastic_distort(image, alpha, sigma):
    random_state = np.random.RandomState(None)
    shape = image.shape

    dx = gaussian_filter(
        (random_state.rand(*shape) * 2 - 1),
        sigma, mode="constant"
    ) * alpha
    dy = gaussian_filter(
        (random_state.rand(*shape) * 2 - 1),
        sigma, mode="constant"
    ) * alpha

    x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]))
    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1))
    return map_coordinates(image, indices, order=1).reshape(shape)


'''
def make_regular_image(src_path, dst_path):
    labels = useful_label()
    for label in labels:
        images = listdir(src_path+"\\t")
        for image_path in images:
            print(src_path+"\\"+label+"\\"+image_path)
            img = cv2.imread(src_path+"\\t\\"+image_path, cv2.IMREAD_COLOR)
            print(img)
            cv2.imshow("tesT", img)
            cv2.waitKey()
            img = preprocessor.image_regular(img)
            cv2.imwrite(dst_path=dst_path+"\\"+label+"\\"+image_path, img=img)
'''


if __name__ == "__main__":
    make_dir(DATA_FOLDER_PATH+"/dataset/font_label")
    make_dataset()

