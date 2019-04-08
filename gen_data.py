from PIL import Image
import os,glob
import numpy as np
from sklearn import model_selection
classes=["monkey","boar","crow"]
num_classes=len(classes)
image_size=50

#画像の読み込み

X=[]
Y=[]

for index, classlabel in enumerate(classes):
    photos_dir="./"+ classlabel
    files=glob.glob(photos_dir+"/*.jpg")
    for i, file in enumerate(files):
        if i>= 200: break
        image=Image.open(file)
        #open() は PIL.Image というモジュールのグローバル関数
        image=image.convert("RGB")
        image=image.resize((image_size,image_size))
        #convert や resize は Image.open() が返す画像オブジェクトが
        #持つメソッドなので、image.convert(), image.resize()
        #と呼び出します
        data=np.asarray(image)
        X.append(data)
        Y.append(index)
X=np.array(X)
Y=np.array(Y)

X_train,X_test,y_train,y_test=model_selection.train_test_split(X,Y)
xy=(X_train,X_test,y_train,y_test)
np.save("./animal.npy",xy)

#activate tf140で実行
