import numpy as np
import optuna
import os
from PIL import Image
import glob
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.callbacks import TensorBoard,ModelCheckpoint
print(tf.version)
from keras.layers import BatchNormalization
# classes = ["text","image","diagram"]
# datadir="/media/sf_virtualbox/My_code/Technical_Knowledge/computer-vision/data"
datadir=""
classes = ["dog","cat"]
num_classes = len(classes)

image_size = 224
# input_shape = (image_size,image_size,2)

def machine_learning_data():
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

    input_shape=X_train.shape[1:]
    print("input_shape",input_shape)
    return X_train, X_test, y_train, y_test, input_shape


def create_model(n_layer,n_filter,activation, filter_step, mid_units, dropout_rate, input_shape):
    model = Sequential()

    for i in range(n_layer):
        print("n_layer",n_layer,"n_filter",n_filter,'stride',filter_step,"activation", activation, "mid_units", mid_units, "dropout_rate",dropout_rate)
        model.add(Conv2D(filters=n_filter, kernel_size=(3, 3), padding='same',strides=(filter_step,filter_step), activation=activation, input_shape=input_shape))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(BatchNormalization())

    model.add(Flatten())
    model.add(Dropout(dropout_rate))

    model.add(Dense(2, activation=activation))

    return model


def objective(trial):
    x_train, x_test, y_train, y_test, input_shape = machine_learning_data()

    n_layer = trial.suggest_int('n_layer', 1, 3) # 追加する層を1-3から選ぶ
    n_filter = trial.suggest_int('n_filter', 16, 64) # 畳み込みフィルターの数
    filter_step = trial.suggest_int('filter_step',1,3) # 畳み込みフィルターのストライド数
    mid_units = trial.suggest_int('mid_units', 10, 100) # ユニット数
    dropout_rate = trial.suggest_uniform('dropout_rate', 0, 0.5) # ドロップアウト率
    activation = trial.suggest_categorical('activation', ['relu','sigmoid']) # 活性化関数
    optimizer = trial.suggest_categorical('optimizer', ['sgd', 'adam', 'rmsprop']) # 最適化アルゴリズム

    # 学習モデルの構築と学習の開始
    model = create_model(n_layer,n_filter,activation, filter_step, mid_units, dropout_rate, input_shape)
    model.compile(optimizer=optimizer,
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])
#     history = model.fit(x_train, y_train, 
#                         verbose=1,
#                         epochs=10,
#                         validation_data=(x_test, y_test),
#                         batch_size=100)
#     log_dir = os.path.join(os.path.dirname(__file__), "logdir")
#     model_file_name="model_file.hdf5"
    history = model.fit(
                        x_train, y_train,
                        epochs=10,
                        validation_split = 0.2
                        #  過学習が起きないようにするためコールバック関数挿入
                        # callbacks=[
                        #         TensorBoard(log_dir=log_dir),
                        #         ModelCheckpoint(os.path.join(log_dir,model_file_name),save_best_only=True)
                        #         ]
                        )
#     model.save_weights('keras_model.hdf5')
    loss,accuracy = model.evaluate(x_test, y_test, verbose=1)

    # print('Test Loss : ', loss)
    # print('Test Accuracy : ', accuracy)
    # plt.plot(history.history['accuracy'])
    # plt.plot(history.history['val_accuracy'])
    # plt.title('Model accuracy')
    # plt.ylabel('Accuracy')
    # plt.xlabel('Epoch')
    # plt.grid()
    # plt.legend(['Train', 'Validation'], loc='upper left')
    # plt.show()
    # plt.savefig("TL_keras.jpg")

    return -np.amax(history.history['val_accuracy'])

def main():
    # TPE 
    # study = optuna.create_study(sampler=optuna.samplers.TPESampler())
    # CMA-ES
    study = optuna.create_study(sampler=optuna.samplers.CmaEsSampler())
    study.optimize(objective, n_trials=100)
    print('best_params')
    print(study.best_params)
    print('-1 x best_value')
    print(-study.best_value)

    sortdict = sorted(study.best_params.items())
    z = np.zeros((len(sortdict),1))
    z = z.astype(np.unicode)
    for i,l in enumerate(sortdict):
        print(i,l)
        z[i,0] = l[0]

#     np.save("best_params_item",z)

    sortdict = sorted(study.best_params.items())
    z = np.zeros((len(sortdict),1))
    z = z.astype(np.unicode)
    for i,l in enumerate(sortdict):
        print(i,l)
        z[i,0] = l[1]

#     np.save("best_params_value",z)

    print('\n --- sorted --- \n')
    sorted_best_params = sorted(study.best_params.items(), key=lambda x : x[0])
    for i, k in sorted_best_params:
        print(i + ' : ' + str(k))


if __name__ == '__main__':
    main()