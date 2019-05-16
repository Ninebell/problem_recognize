
from module.preprocessor import *
from model.Hangul import *
from module.parser import *


if __name__ == "__main__":
    img_path = img_path()
    #img = cv_imload(img_path)
    img = sample_img()
    cv_imshow("input", img, 0)
    words = split_img_to_words(img)

    model = load_model("model/model1.h5")
    labels = useful_label()
    text = ""
    for word in words:
        cv_imshow("img", word, 0)
        info = np.asarray(word)
        info = info.reshape([1, 64, 64, 1])
        info = info / 255.0
        predict = model.predict(info)
        idx = np.argmax(predict)
        if predict[0][idx] > 0.95:
            print("value: {}, labels: {}".format(predict[0][idx], labels[idx]))
            text = text + labels[idx]
        else:
            text = text + " "

    print(text)
    cv_imshow("input", img, 0)

