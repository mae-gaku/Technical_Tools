from PIL import Image
import os, glob
import numpy as np
from keras.utils import np_utils
from sklearn import model_selection
from sklearn.model_selection import train_test_split


#判別するラベル
classes =  ["image", "diagram","text"]
num_classes = len(classes)
# image_size = 128
image_size = 224


datadir=''

#画像の読み込み
X = []
Y = []

# 水増し

for index, classlabel in enumerate(classes):
    photos_dir = datadir+ classlabel
    files = glob.glob(photos_dir + "/*.jpg")
    for i, file in enumerate(files):

        image = Image.open(file)
        image = image.convert("RGB")
        image = image.resize((image_size, image_size))
        #image.save("./test/{}{}.jpg".format(classlabel,i))
        data = np.asarray(image)

        for angle in range(-10, 10, 10):
            # 回転
            # img_r = image.rotate(angle)
            # data = np.asarray(img_r)
            # X.append(data)
            # Y.append(index)

            # 反転
            img_trans = image.transpose(Image.FLIP_LEFT_RIGHT)
            data = np.asarray(img_trans)
            X.append(data)
            Y.append(index)



X = np.array(X)
Y = np.array(Y)

#２割テストデータへ
(X_train, X_test, y_train, y_test) = train_test_split(X, Y,test_size=0.2)

#正規化
X_train = X_train.astype("float") / 255
X_test = X_test.astype("float") / 255

#教師データの型を変換
y_train = np_utils.to_categorical(y_train,num_classes)
y_test = np_utils.to_categorical(y_test, num_classes)
# y_train = np_utils.to_categorical(y_train)
# y_test = np_utils.to_categorical(y_test)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

#X_train, X_test, y_train, y_test = model_selection.train_test_split(X, Y)
xy = (X_train, X_test, y_train, y_test)
np.save("./dataset.npy", xy)


