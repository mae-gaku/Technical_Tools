from PIL import Image
import os, glob
import numpy as np
from keras.utils import np_utils
from keras.layers import BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,GlobalAveragePooling2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras.optimizers import RMSprop,SGD
from tensorflow.keras.utils import to_categorical
from tensorflow import keras
from efficientnet.tfkeras import EfficientNetB0
from tensorflow.keras.applications import EfficientNetB0
from keras.callbacks import TensorBoard,ModelCheckpoint
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
from tensorflow.keras import optimizers
from tensorflow.keras import backend as K
import numpy as np
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from PIL import Image
import os
import cv2
import matplotlib.pyplot as plt

hp1 = {}
hp1['class_num'] = 2  
hp1['batch_size'] = 16
hp1['epoch'] = 5

classes = ["dog","cat"]
num_classes = len(classes)
image_size = 224

datadir=""

X = []
Y = []

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
            img_r = image.rotate(angle)
            data = np.asarray(img_r)
            X.append(data)
            Y.append(index)

            img_trans = image.transpose(Image.FLIP_LEFT_RIGHT)
            data = np.asarray(img_trans)
            X.append(data)
            Y.append(index)

X = np.array(X)
Y = np.array(Y)

(X_train, X_test, y_train, y_test) = train_test_split(X, Y,test_size=0.2)

X_train = X_train.astype("float") / 255
X_test = X_test.astype("float") / 255

y_train = np_utils.to_categorical(y_train,num_classes)
y_test = np_utils.to_categorical(y_test, num_classes)
# y_train = np_utils.to_categorical(y_train)
# y_test = np_utils.to_categorical(y_test)

image_size = 224
input_shape = (image_size,image_size,3)

# model
from efficientnet.tfkeras import EfficientNetB0

def CNN(input_shape):
        eff = EfficientNetB0(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape)
        x = tf.keras.layers.GlobalAveragePooling2D()(eff.output)
        output = tf.keras.layers.Dense(2, activation='softmax', name='last_output')(x)
        model = tf.keras.Model(inputs=eff.inputs, outputs=output, name='model')

        return model

model=CNN(input_shape)

model.compile(loss='categorical_crossentropy',optimizer='SGD',metrics=['accuracy'])

log_dir = os.path.join(os.path.dirname(__file__), "logdir")
model_file_name="model_file.hdf5"

history = model.fit(
        X_train, y_train,
         epochs=hp1['epoch'],
         validation_split = 0.2,
         callbacks=[
                TensorBoard(log_dir=log_dir),
                ModelCheckpoint(os.path.join(log_dir,model_file_name),save_best_only=True)
                ]
        )

loss,accuracy = model.evaluate(X_test, y_test, batch_size=hp1['batch_size'])

# create heatmap
n = 10
preprocessed_input = X_test[n:n+1].copy()
predictions = model.predict(preprocessed_input)
cls = np.argmax(predictions)
layer_name='top_activation'

# grad cam
y_c = model.output[0, cls]
conv_output = model.get_layer(layer_name).output
g = tf.Graph()

with g.as_default():
    grads = tf.gradients(y_c, conv_output)[0]

gradient_function = K.function([model.input], [conv_output, grads])
output, grads_val = gradient_function([preprocessed_input])
output, grads_val = output[0, :], grads_val[0, :, :, :]
weights = np.mean(grads_val, axis=(0, 1))
cam = np.dot(output, weights)

# Process CAM
cam = cv2.resize(cam, (image_size, image_size), cv2.INTER_LINEAR)
cam = np.maximum(cam, 0)
cam = cam / cam.max()
gradcam = cam

labels = {
    0:"dog",
    1:"cat",
}

# visualise grad cam
print("TRUE : {}".format(labels[y_test[n,0]]))
print("PRED : {}".format(labels[cls]))
for i in range(hp1['class_num']):
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
plt.savefig("image1.jpg")
