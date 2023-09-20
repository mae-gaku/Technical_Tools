from PIL import Image
import os, glob
import numpy as np
from keras.utils import np_utils
from sklearn import model_selection
from sklearn.model_selection import train_test_split

import glob
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image,ImageOps
import sklearn
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers
from IPython.display import display
from sklearn.utils import shuffle
from matplotlib import rcParams
import warnings
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
import seaborn as sns

from tensorflow.keras.layers import Activation,Dropout,Flatten,Dense,GlobalAveragePooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop,SGD
from tensorflow.keras.utils import to_categorical
from tensorflow import keras
from tensorflow.keras.applications import EfficientNetV2B0
from keras.callbacks import TensorBoard,ModelCheckpoint
import tensorflow as tf
from tensorflow.keras import optimizers
import tensorflow
from tensorflow.keras import optimizers
from keras.utils import np_utils
from keras.layers import BatchNormalization
from matplotlib import image
# from keras.optimizers import RMSprop,SGD
# import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D

#判別するラベル
classes = ["cloudy","desert","green_area","water"]
num_classes = len(classes)
# image_size = 128
image_size = 256


datadir=""

#画像の読み込み
X = []
Y = []

# 水増し

for index, classlabel in enumerate(classes):
    photos_dir = datadir+ classlabel
    files = glob.glob(photos_dir + "/*")
    for i, file in enumerate(files):

        image = Image.open(file)
        image = image.convert("RGB")
        image = image.resize((image_size, image_size))
        #image.save("./test/{}{}.jpg".format(classlabel,i))
        data = np.asarray(image)

        for angle in range(0, 10, 10):
            # 回転
            img_r = image.rotate(angle)
            data = np.asarray(img_r)
            X.append(data)
            Y.append(index)

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



image_size = 256
input_shape = (image_size,image_size,3)


# CNNを構築
def CNN(input_shape):
        model = Sequential()
        model.add(EfficientNetV2B0(
                include_top=False,
                weights='imagenet',
                input_shape=input_shape))

        model.add(GlobalAveragePooling2D())

        model.add(Dense(4))
        model.add(Activation('softmax'))

        return model

model=CNN(input_shape)

# optimization set up
opt = tensorflow.keras.optimizers.RMSprop(lr=0.0001, decay=1e-6)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])


log_dir = os.path.join(os.path.dirname('__file__'), "logdir")
model_file_name="model_file.hdf5"

#訓練
history = model.fit(
        X_train, y_train,
        batch_size = 32,
         epochs=100,
         validation_split = 0.2,
        # #  過学習が起きないようにするためコールバック関数挿入
         callbacks=[
                TensorBoard(log_dir=log_dir),
                ModelCheckpoint(os.path.join(log_dir,model_file_name),save_best_only=True)
                ]
        )

#評価 & 評価結果出力

loss,accuracy = model.evaluate(X_test, y_test, verbose=1)

print('Test Loss : ', loss)
print('Test Accuracy : ', accuracy)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.grid()
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()