# 使えなかったためtensorflow.kerasを使用した。
# import keras
# from keras.utils import np_utils
# from keras.models import Sequential
# from keras.layers.convolutional import Conv2D, MaxPooling2D
# from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import BatchNormalization
from matplotlib import image
# from keras.optimizers import RMSprop,SGD
# import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras.optimizers import RMSprop,SGD
from tensorflow.keras.utils import to_categorical
from tensorflow import keras
import numpy as np
from sklearn.model_selection import train_test_split
from PIL import Image
import glob

from PIL import Image
import matplotlib.pyplot as plt

import os
from keras.callbacks import TensorBoard,ModelCheckpoint



#ハイパーパラメーター
hp1 = {}
# クラス数
hp1['class_num'] = 3 
# バッチサイズ 
hp1['batch_size'] = 64
#エポック数
hp1['epoch'] = 20 



#データセットのロード
#1で作ったデータセットをここで読み込む
X_train, X_test, y_train, y_test = np.load("", allow_pickle=True)


#入力サイズ
input_shape=X_train.shape[1:]

# CNNを構築
def CNN(input_shape):

        # モデル層を積み重ねる形式の記述。
        # .addメソッドで追加できる。
        model = Sequential()

        # Cnv2D：2次元畳み込み層。空間フィルター畳み込み演算層
        # Conv2D(32,(3,3))：「3×3」の大きさのフィルタを32枚使う。
        # ※フィルタは「5×5」「7×7」などと、中心を求められる奇数が使いやすい。
        # フィルタ数は、「16・32・64・128・256・512」などが使われる傾向にあるが、複雑な場合はフィルタ数を多めにする。
        # padding='same'：出力画像のサイズが変わらないようにするためpaddingを実施。sameを指定すると、画像の周りを0パディングすることによってサイズを変えず端の特徴もより捉えることができる。
        # input_shape：例）input_shape=(28, 28, 1)　→　縦28・横28ピクセルのグレースケールを入力する＋チャネル数。※色の情報を持ていれば「3」となる。(入力の形状)
        model.add(Conv2D(32, (3, 3), padding='same',input_shape=input_shape))
        # activation=’relu’：活性化関数「ReLU（ランプ関数）」。フィルタ後の画像に実施。
        # 入力が0以下の時は出力0。入力が0より大きい時はそのまま出力する。
        model.add(Activation('relu'))

        model.add(Conv2D(32, (3, 3)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))

        # 「2×2」の大きさの最大プーリング層。入力画像内の「2×2」の領域で最大の数値を出力する
        model.add(MaxPooling2D(pool_size=(2, 2)))

        # ドロップアウト：過学習予防。
        # 全結合の層とのつながりを「25％」無効化している
        model.add(Dropout(0.25))


        
        model.add(Conv2D(64, (3, 3), padding='same'))
        model.add(Activation('relu'))

        model.add(Conv2D(64, (3, 3)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(128, (3, 3), padding='same'))
        model.add(Activation('relu'))
        
        model.add(Conv2D(128, (3, 3)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        # 平坦化(次元削減)→一次元ベクトルに変換する
        model.add(Flatten())
        # 全結合層。出力512。
        model.add(Dense(512))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        # image.text,diargramの3つを分類するためDense(3)となる
        # 出力が3つ
        model.add(Dense(hp1['class_num']))
        # softmax関数：確率の結果の総和を1にする
        model.add(Activation('softmax'))

        return model


#モデルを選択
model=CNN(input_shape)



#コンパイル

# SGD = RMSprop(lr=0.00005, decay=1e-6)

# オプティマイザ(optimizer)：トレーニングを最適化する手法を設定する。
# 損失関数：ラベルデータと実際の出力どれだけ誤差があるのかを計算する関数
# categorical_crossentropyを採用。
# ラベルを1-of-K表現に変形せずともsparse_categorical_crossentropyを使えばそのままの形状で渡すことが可能
model.compile(loss='categorical_crossentropy',optimizer='SGD',metrics=['accuracy'])

# model.compile(loss='categorical_crossentropy',optimizer= 'SGD',metrics=['accuracy'])

#データの記録
log_dir = os.path.join(os.path.dirname(__file__), "logdir")
model_file_name="model_file.hdf5"

print(X_train.shape)
print(y_train.shape)
#訓練
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

#評価 & 評価結果出力

loss,accuracy = model.evaluate(X_test, y_test, batch_size=hp1['batch_size'])


