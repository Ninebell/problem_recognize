from os import listdir
from os.path import isfile, join

import random

import cv2

import numpy as np
from glob import glob
from PIL import Image

import tensorflow as tf
from keras.utils import to_categorical
from keras.models import model_from_json

from module.maker import useful_label, get_fonts, DATA_FOLDER_PATH

PATH_MODEL = "C:/Users/Jonghoe/PycharmProjects/math_expression/model"

IMAGE_WIDTH = 64
IMAGE_HEIGHT = 64
CLASS_NUM = 2386


def load_dataset(view=True):
    entire_image = []
    entire_label = []

    dataset_path = DATA_FOLDER_PATH + "/dataset/font_label"
    han_word_paths = listdir(dataset_path)
    total = 0
    for w_num, word_path in enumerate(han_word_paths):
        image_paths = glob(join(dataset_path, word_path, '*.png'))
        total += len(image_paths)

    count = 0
    per = total//100
    for w_num, word_path in enumerate(han_word_paths):
        image_paths = glob(join(dataset_path, word_path, '*.png'))

        for image_path in image_paths:
            entire_image.append(np.asarray(Image.open(image_path)))
            entire_label.append(to_categorical(w_num, num_classes=CLASS_NUM))    # to_categorial = one_hot_encoding
            count = count + 1
            if count % per == 0 and view:
                print("{}/{} {}> {}%".format(count, total, '='*(count//per), count//per))

    return (entire_image, entire_label)


def make_model(image_size, class_num):

    model = tf.keras.models.Sequential()
    # input 64 * 64
#    model.add(tf.keras.layers.InputLayer(image_size))

    # conv
    model.add(tf.keras.layers.Conv2D(32, (3, 3), padding='same', input_shape=image_size,activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D())

    model.add(tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D())

    model.add(tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D())

    model.add(tf.keras.layers.Flatten())

    model.add(tf.keras.layers.Dense(1024, activation='relu'))

    model.add(tf.keras.layers.Dense(class_num, activation='softmax'))

    model.summary()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def divide_train_test(x_entire, y_entire, ratio, font_count):

    train_len = (int)(len(x_entire) * ratio)
    train_ratio = (int)(font_count * ratio)
    print(train_ratio)
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    idx = []

    for i in range(0, font_count):
        idx.append(i)

    i = 0
    print("{}, {}".format(len(x_entire), font_count))

    random.shuffle(idx)
    while i < len(x_entire):
        for j in range(0, font_count):
            if j % font_count < train_ratio:
                x_train.append(x_entire[i+idx[j]])
                y_train.append(y_entire[i+idx[j]])

            else:
                x_test.append(x_entire[i+idx[j]])
                y_test.append(y_entire[i+idx[j]])

        i = i + font_count

    x_train = np.asarray(x_train)
    y_train = np.asarray(y_train)
    x_test = np.asarray(x_test)
    y_test = np.asarray(y_test)

    x_train = x_train.reshape(x_train.shape[0], IMAGE_HEIGHT, IMAGE_WIDTH, 1)
    x_test = x_test.reshape(x_test.shape[0], IMAGE_HEIGHT, IMAGE_WIDTH, 1)
    return (x_train, y_train), (x_test, y_test)


def load_model(model_path):
    print(model_path)
    return tf.keras.models.load_model(model_path)
#        return loaded_model_json

    '''
    print(model_path)
    loaded_model_json = tf.keras.models.load_model(model_path)
    '''


def get_dataset():
    fonts = get_fonts()
    entire_img, entire_label = load_dataset()

    (x_train, y_train), (x_test, y_test) = divide_train_test(entire_img, entire_label, 0.7, len(fonts))

    x_train = x_train/255.0
    x_test = x_test/255.0

    return (x_train, y_train), (x_test, y_test)


def train_model(epoch, batch_size):
    (x_train, y_train), (x_test, y_test) = get_dataset()
    model = make_model((IMAGE_HEIGHT, IMAGE_WIDTH, 1), CLASS_NUM)
    model.fit(x_train, y_train, epochs=epoch, batch_size=batch_size, shuffle=True)

    loss_metrics = model.evaluate(x_test, y_test)
    print("========================")
    print(loss_metrics)

    return model


if __name__ == "__main__":
    model = train_model(30, 100)
    print("save_model")
    model.save("all_char_model.h5")
    print("========================")


