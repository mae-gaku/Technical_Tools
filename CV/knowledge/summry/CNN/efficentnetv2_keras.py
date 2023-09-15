from xml.etree.ElementInclude import include

from pydicom import Sequence
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Flatten, Dense, Input,Activation
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import optimizers,models,layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
# EfficientNetV2B0 model
from tensorflow.keras.applications import EfficientNetV2B0

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

from keras.callbacks import TensorBoard,ModelCheckpoint


train_images_path =  ""
train_images_path = list(paths.list_images(train_images_path))

data = []
labels = []

for imagePath in train_images_path:
    label = imagePath.split(os.path.sep)[-2]
    image = cv2.imread(imagePath)
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    imgae = cv2.resize(image,(256,256))
    data.append(image)
    labels.append(label)


data = np.array(data) / 255.0
labels = np.array(labels)

label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(labels)
labels = to_categorical(integer_encoded)

print(".....",len(data),'Image loaded in 3x classes:')
print(label_encoder.classes_)

(trainX, testX, trainY, testY)  = train_test_split(data, labels, test_size = 0.20, stratify=labels,random_state=42)


image_size = 256
input_shape = (image_size,image_size,3)

def model_cnn(input_shape):
    model = Sequential()
    model.add(EfficientNetV2B0(include_top=False,weights='imagenet',input_shape=input_shape))
    model.add(GlobalAveragePooling2D())
    model.add(Dense(3))
    model.add(Activation('softmax'))
        
    return model

model = model_cnn(input_shape)

model.compile(loss='categorical_crossentropy',optimizer='SGD',metrics=['accuracy'])


log_dir = os.path.join(os.path.dirname(__file__), "logdir")
model_file_name="model_file.hdf5"

#訓練
history = model.fit(
        trainX, trainY,
         epochs=30,
         validation_split = 0.2,
        #  過学習が起きないようにするためコールバック関数挿入
         callbacks=[
                TensorBoard(log_dir=log_dir),
                ModelCheckpoint(os.path.join(log_dir,model_file_name),save_best_only=True)
                ]
        )

#評価 & 評価結果出力

loss,accuracy = model.evaluate(testX, testY, batch_size=32)
