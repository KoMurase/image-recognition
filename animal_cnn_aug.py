#cnn　畳み込みニューラルネットワーク
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D
from keras.layers import Activation,Dropout,Flatten,Dense
#Activation:活性化関数 Dropout:ドロップアウトを行う関数
#Flatten:データを一次限にする関数
from keras.utils import np_utils
import numpy as np
import keras

classes=["monkey","boar","crow"]
num_classes=len(classes)
image_size=50

#メインの関数を定義する
def main():
    X_train,X_test,y_train,y_test=np.load("./animal_aug.npy")
    #正規化する
    X_train=X_train.astype("float")/256
    y_train=y_train.astype("float")/256
    #one-hot-vector:正解は1,他は0
    y_train=np_utils.to_categorical(y_train,num_classes)
    y_test=np_utils.to_categorical(y_test,num_classes)

    model=model_train(X_train,y_train)
    model_eval(model,X_test,y_test)

def model_train(X,y):
    model=Sequential()
    #padding='same'で畳み込み結果が同じサイズになるようにピクセルを合わせる
    #input_shapeは入力画像の形状
    model.add(Conv2D(32,(3,3),padding='same',input_shape=X.shape[1:]))
    #reluは負の値を0にする
    model.add(Activation('relu'))
    model.add(Conv2D(32,(3,3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64,(3,3),padding='same'))
        #reluは負の値を0にする
    model.add(Activation('relu'))
    model.add(Conv2D(64,(3,3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(3))
    #softmax:画像の一致しているところを１としている
    model.add(Activation('softmax'))

    #decayで学習率を下げる
    opt=keras.optimizers.rmsprop(lr=0.0001,decay=1e-6)

    model.compile(loss='categorical_crossentropy',
    optimizer=opt,metrics=['accuracy'])

    model.fit(X,y,batch_size=32,epochs=100)

    #モデルの保存
    model.save('./animal_cnn_aug.h5')

    return model

def model_eval(model,X,y):
    scores=model.evaluate(X,y,verbose=1)
    print('Test Loss:',scores[0])
    print('Test Accuracy:',scores[1])

    #もしこのプログラムが直接呼ばれたらmainを呼ぶ
if __name__=="__main__":
    main()
