# from imp import load_module
# from msvcrt import kbhit
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.preprocessing.image import array_to_img, img_to_array,ImageDataGenerator
import numpy as np


def drop_resolution(x,scale):
   #画像のサイズをいったん縮小したのちに再び拡大することで、低解像度の画像を作る。
   size=(x.shape[0],x.shape[1])
   small_size=(int(size[0]/scale),int(size[1]/scale))
   img=array_to_img(x)
   small_img=img.resize(small_size, 3)
   return img_to_array(small_img.resize(img.size,3))

def data_generator(data_dir,mode,scale,target_size=(256,256),batch_size=32,shuffle=True):
   for imgs in ImageDataGenerator().flow_from_directory(
       directory=data_dir,
       classes=[mode],
       class_mode=None,
       color_mode='rgb',
       target_size=target_size,
       batch_size=batch_size,
       shuffle=shuffle
   ):
       x=np.array([
           drop_resolution(img,scale) for img in imgs
           ])
       yield x/255.,imgs/255.


DATA_DIR='/home/gaku/CNN/images/'
N_TRAIN_DATA=151#学習用データ数
N_TEST_DATA=30 #評価用データ数
BATCH_SIZE=32 #バッチサイズ

train_data_generator=data_generator(
   DATA_DIR,'train', scale=4.0, batch_size=BATCH_SIZE
)

test_x,test_y=next(
   data_generator(
   DATA_DIR,'test',scale=4.0, batch_size=BATCH_SIZE,shuffle=False
   )
)


model = Sequential()
input_shape = (256,256,3)


model.add(Conv2D(filters=64, kernel_size= 9, activation='relu', padding='same', input_shape=input_shape))
model.add(Conv2D(filters=32, kernel_size= 1, activation='relu', padding='same'))
model.add(Conv2D(filters=3, kernel_size= 5, padding='same'))

model.summary()

def psnr(y_true,y_pred):
    return -10*K.log(K.mean(K.flatten((y_true-y_pred))**2)
    )/np.log(10)


model.compile(
    loss='mean_squared_error',
    optimizer= 'adam',
    metrics=[psnr]
)

model.fit_generator(
    train_data_generator,
    validation_data=(test_x,test_y),
    steps_per_epoch=N_TRAIN_DATA//BATCH_SIZE,
    epochs=10
)

