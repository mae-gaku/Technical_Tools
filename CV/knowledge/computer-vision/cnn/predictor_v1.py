#ライブラリインポート
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Flatten

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from efficientnet.tfkeras import EfficientNetB0
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import cifar10

import datetime

from tensorflow.keras import backend as K
import cv2
import os
from keras.callbacks import TensorBoard,ModelCheckpoint

#ハイパーパラメーター
hp1 = {}
# クラス数
hp1['class_num'] = 2 
# バッチサイズ 
hp1['batch_size'] = 64
#エポック数
hp1['epoch'] = 20 

# データインポート
X_train, X_test, y_train, y_test = np.load("", allow_pickle=True)

# 画像リサイズ
image_size  = 224
input_shape = (image_size,image_size,3)


# 予測するクラスの数
num_classes = 2

# モデル定義
def build_model():
    eff = EfficientNetB0(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape)
    x = tf.keras.layers.GlobalAveragePooling2D()(eff.output)
    output = tf.keras.layers.Dense(num_classes, activation='softmax', name='last_output')(x)
    model = tf.keras.Model(inputs=eff.inputs, outputs=output, name='model')

    model.compile(optimizer=optimizers.Adam(learning_rate=1e-4), loss="categorical_crossentropy", metrics=["accuracy"])
    
    return model

#モデル構築
model = build_model()

log_dir = os.path.join(os.path.dirname(__file__), "logdir")
model_file_name="model_file.hdf5"
# 学習
history = model.fit(
        X_train, y_train,
         epochs=hp1['epoch'],
         validation_split = 0.2,
        #  過学習が起きないようにするためコールバック関数挿入
         callbacks=[
                TensorBoard(log_dir=log_dir),
                ModelCheckpoint(os.path.join(log_dir,model_file_name),save_best_only=True)
                ]
        )

loss,accuracy = model.evaluate(X_test, y_test, batch_size=hp1['batch_size'])


# grad cam関数
def grad_cam(input_model, image, cls, layer_name):
    """GradCAM method for visualizing input saliency."""
    y_c = input_model.output[0, cls]
    conv_output = input_model.get_layer(layer_name).output
    g = tf.Graph()
    with g.as_default():
        grads = tf.gradients(y_c, conv_output)[0]
    # Normalize if necessary
    # grads = normalize(grads)
    gradient_function = K.function([input_model.input], [conv_output, grads])

    output, grads_val = gradient_function([image])
    output, grads_val = output[0, :], grads_val[0, :, :, :]

    weights = np.mean(grads_val, axis=(0, 1))
    cam = np.dot(output, weights)

    # Process CAM
    cam = cv2.resize(cam, (image_size, image_size), cv2.INTER_LINEAR)
    cam = np.maximum(cam, 0)
    cam = cam / cam.max()
    return cam

# modelの確認
model.summary()

# cifar10 class labels
labels = {
    0:"dog",
    1:"cat",
   
}

# grad cam
n = 16 #n番目の画像に対してGradCAM適用する
preprocessed_input = X_test[n:n+1].copy()
predictions = model.predict(preprocessed_input)
cls = np.argmax(predictions)
layer_name='top_activation'

gradcam = grad_cam(model, preprocessed_input, cls, layer_name)

# visualise grad cam
print("TRUE : {}".format(labels[y_test[n,0]]))
print("PRED : {}".format(labels[cls]))
for i in range(num_classes):
    print("  {:<10s} : {:5.2f}%".format(labels[i],predictions[0,i]*100))

plt.figure(figsize=(5, 5))
plt.subplot(121)
plt.title('GradCAM')
plt.imshow(preprocessed_input[0])
plt.imshow(gradcam, cmap='jet', alpha=0.5)

plt.subplot(122)
plt.title('Original')
plt.imshow(preprocessed_input[0])

plt.show()