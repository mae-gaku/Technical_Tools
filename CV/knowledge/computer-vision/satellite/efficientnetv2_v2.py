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



def loaddata():
    dataset = ""

    label_names = ["cloudy","desert","green_area","water"]

    images = []
    labels = []

    for folder in os.listdir(dataset):
        files = glob.glob(pathname=str(dataset +folder + "/*.jpg"))

        label = label_names.index(folder)
        for file in files:
            image = cv2.imread(file)
            image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
            image = cv2.resize(image,(256,256))

            images .append(image)
            labels.append(label)
    
    images = np.array(images)
    labels = np.array(labels)

    return images,labels
            

images,labels = loaddata()

labels = np.array(pd.get_dummies(labels))
(X_train, X_test, Y_train, Y_test) = train_test_split(images, labels, test_size=0.20)

# X_train = X_train.astype("float") / 255
# X_test = X_test.astype("float") / 255


print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)


input_shape = (256,256,4)
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

model = CNN(input_shape)

# optimization set up
opt = tensorflow.keras.optimizers.RMSprop(lr=0.0001, decay=1e-6)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
 
log_dir = os.path.join(os.path.dirname(__file__), "logdir")
model_file_name="model_file.hdf5"

# # training
# history = model.fit(X_train, Y_train, batch_size=32, epochs=30)
#訓練
history = model.fit(
        X_train, Y_train,
         epochs=30,
         validation_split = 0.2,
        #  過学習が起きないようにするためコールバック関数挿入
         callbacks=[
                TensorBoard(log_dir=log_dir),
                ModelCheckpoint(os.path.join(log_dir,model_file_name),save_best_only=True)
                ]
        )
#評価 & 評価結果出力

loss,accuracy = model.evaluate(X_test, Y_train, batch_size=32)

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