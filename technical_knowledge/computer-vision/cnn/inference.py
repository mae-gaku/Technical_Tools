from pathlib import Path
import numpy as np
from PIL import Image
from keras.models import load_model
import glob
import shutil
import os

from tensorflow.keras import backend, layers


class FixedDropout(layers.Dropout):
        def _get_noise_shape(self, inputs):
            if self.noise_shape is None:
                return self.noise_shape

            symbolic_shape = backend.shape(inputs)
            noise_shape = [symbolic_shape[axis] if shape is None else shape
                           for axis, shape in enumerate(self.noise_shape)]
            return tuple(noise_shape)

 
#学習済みモデルの読み込み
model_path = ""

#予測したいデータのフォルダ
images_folder = ""

classes =  ["dog", "cat"]

model = load_model(model_path,custom_objects={'FixedDropout':FixedDropout(rate=0.2)})


image_size=224
X = []


dir = images_folder
#パスの確認
#print(dir)

files = glob.glob(dir + "/*.jpg")
for i, file in enumerate(files):
    image = Image.open(file)
    image = image.convert("RGB")
    image = image.resize((image_size, image_size))
    data = np.asarray(image)
    X.append(data)


X = np.array(X)


#正規化する
X = X.astype('float32')
X = X / 255.0

#print(len(files))

#softmax
for w in range(len(files)):

    result = model.predict([X])[w]
    predicted = result.argmax()
    percentage = int(result[predicted] * 100)

    print(files[w].split('\\')[-1])
    print("{0}({1} %)".format(classes[predicted],percentage))
    # print(predicted)
    # print(files[w])


    

    


