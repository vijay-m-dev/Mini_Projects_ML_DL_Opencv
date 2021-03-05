#Importing Libraries
import numpy as np
#import pandas as pd 
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
import os
import cv2
from sklearn import preprocessing

train_path = []
label_train = []
test_path = []
label_test = []
X_train = []
X_test = []

#Extracting path and labels of images
path_train = "./fingers/train/"
for filename in os.listdir(path_train):
    train_path.append(path_train+filename)
    whole_label = filename.split('_')[1]
    useful_label = whole_label.split('.')[0]
    label_train.append(useful_label)

path_test = "./fingers/test/"
for filename in os.listdir(path_test):
    test_path.append(path_test+filename)
    whole_label = filename.split('_')[1]
    useful_label = whole_label.split('.')[0]
    label_test.append(useful_label)

# reading images for train data
for path in train_path:
    image=cv2.imread(path)        
    image=cv2.resize(image,(50,50))
    image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    X_train.append(image)
    
# reading images for test data
for path in test_path:
    image=cv2.imread(path)        
    image=cv2.resize(image,(50,50))
    image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    X_test.append(image)

#image preprocessing
X_test = np.array(X_test)
X_train = np.array(X_train)
print("Shape of X_train: ", X_train.shape)
print("Shape of X_test: ", X_test.shape)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train /= 255
X_test /= 255

#OneHot encoding the dependent variable
lable_encoder = preprocessing.LabelEncoder()
y_train_temp = lable_encoder.fit_transform(label_train)
y_test_temp = lable_encoder.fit_transform(label_test)
y_train = keras.utils.to_categorical(y_train_temp,12)
y_test = keras.utils.to_categorical(y_test_temp,12)

#Reshaping the training and testing images
X_train=X_train.reshape(len(X_train),50,50,1)
X_test=X_test.reshape(len(X_test),50,50,1)

#CNN Sequential model
cnn=Sequential()
#Convolution layers
cnn.add(Conv2D(64,(3,3),padding='same',input_shape=(50,50,1),activation="relu"))
cnn.add(Conv2D(64,(3,3),activation="relu"))
#Pooling layer
cnn.add(MaxPooling2D(pool_size=(2,2)))
#Dropout layer to avoid overfitting
cnn.add(Dropout(0.25))
#Flatten layer
cnn.add(Flatten())
cnn.add(Dense(128,activation="relu"))
cnn.add(Dropout(0.5))
cnn.add(Dense(12,activation="softmax"))
cnn.compile(loss='categorical_crossentropy',optimizer="adam",metrics=['accuracy'])
cnn.summary()
#Training the model
model=cnn.fit(X_train,y_train,batch_size=50,epochs=5,validation_split=0.2,shuffle=True)
#Saving the model
#cnn.save("fingers_detect1.h5")

#Visualizing the accuraacy and Loss
fig,axs=plt.subplots(1,2,figsize=[10,5])

axs[0].plot(model.history['accuracy'],label='train',color="red")
axs[0].plot(model.history['val_accuracy'],label='validation',color="blue")
axs[0].set_title('Model accuracy')
axs[0].legend(loc='upper left')
axs[0].set_ylabel('accuracy')
axs[0].set_xlabel('epoch')

axs[1].plot(model.history['loss'],label='train',color="red")
axs[1].plot(model.history['val_loss'],label='validation',color="blue")
axs[1].set_title('Model loss')
axs[1].legend(loc='upper left')
axs[1].set_xlabel('epoch')
axs[1].set_ylabel('loss')

plt.show()

#Accuracy of testing data
score,accuracy=cnn.evaluate(X_test,y_test)
print('Test score achieved: ',score)
print('Test accuracy achieved: ',accuracy)
pred=cnn.predict(X_test)

#Single image prediction
img=cv2.imread('./fingers/single_prediction/3L.png')
img=cv2.resize(img,(50,50))
img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
img=np.array(img)
img=img.astype('float32')
img/=255
img=img.reshape(1,50,50,1)
pred_single=cnn.predict(img)
result=np.argsort(pred_single[0])[::-1]
result=lable_encoder.inverse_transform([result[0]])
print(result)