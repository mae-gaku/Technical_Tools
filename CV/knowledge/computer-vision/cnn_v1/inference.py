from pathlib import Path
import numpy as np
from PIL import Image
from keras.models import load_model
import glob
import shutil
import os


#学習済みモデルの読み込み
model_path = ""

#予測したいデータのフォルダ
images_folder = ""

classes =  ["image", "diagram","text"]

model = load_model(model_path)


image_size=128
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

    # 各ディレクトリに保存するプログラム
    # image
    if predicted == 0:
        new_path = shutil.move(files[w], 'data/data1/image')
        # print('image')

        # patrn
        # image1 = os.rename(dir +'extracted_'+ "/*.jpg")
        # new_path = shutil.move(image1, 'data1/image')

        # dst = "extracted_" + str(w) + ".jpg"
        # image = os.rename(files[w], dir + '\\' + dst)
        # new_path = shutil.move(image[w], 'data1/image')




        
    # diagram
    elif predicted == 1:
        new_path = shutil.move(files[w], 'data/data1/diagram')
        # print('diagram')

    # text
    elif predicted == 2:
        new_path = shutil.move(files[w], 'data/data1/text')
        # print('text')


    print(files[w].split('\\')[-1])
    print("{0}({1} %)".format(classes[predicted],percentage))
    # print(predicted)
    # print(files[w])


    

    


