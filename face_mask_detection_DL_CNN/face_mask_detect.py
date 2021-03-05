#importing libraries
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D,Dense,MaxPool2D,Flatten
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing import image
tf.__version__

#Training and testing dataset
train_datagen=ImageDataGenerator(rescale=1./255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)
training_set=train_datagen.flow_from_directory('data/train',target_size=(64,64),batch_size=32,class_mode='binary')
test_datagen=ImageDataGenerator(rescale = 1./255)
test_set=test_datagen.flow_from_directory('data/test',target_size=(64,64),batch_size=32,class_mode='binary')


#CNN model
cnn = tf.keras.models.Sequential()
cnn.add(Conv2D(filters=32,padding="same",kernel_size=3,activation='relu',strides=2,input_shape=[64, 64, 3]))
cnn.add(MaxPool2D(pool_size=2,strides=2))
cnn.add(Conv2D(filters=32,padding='same',kernel_size=3,activation='relu'))
cnn.add(MaxPool2D(pool_size=2,strides=2))
cnn.add(Flatten())
cnn.add(Dense(units=128,activation='relu'))
cnn.add(Dense(units=1,activation='sigmoid'))
cnn.compile(optimizer='adam',loss='squared_hinge',metrics=['accuracy'])

cnn.summary()

#Fitting the model
r=cnn.fit(x = training_set,validation_data=test_set,epochs=50)

# plot the loss
import matplotlib.pyplot as plt
plt.plot(r.history['loss'],label='train loss')
plt.plot(r.history['val_loss'],label='val loss')
plt.legend()
plt.show()
#plt.savefig('LossVal_loss')

# plot the accuracy
plt.plot(r.history['accuracy'],label='train acc')
plt.plot(r.history['val_accuracy'],label='val acc')
plt.legend()
plt.show()
#plt.savefig('AccVal_acc')

#Saving and loading model
cnn.save('face_mask_detect.h5')
model = load_model('face_mask_detect.h5')
model.summary()

#testing the model with image
test_image = image.load_img('data/single_test/img_0_1.jpg',target_size=(64,64))
test_image = image.img_to_array(test_image)
test_image=test_image/255
test_image = np.expand_dims(test_image,axis = 0)
result = model.predict(test_image)
print(result[0][0])
if result[0][0]<0.5:
    print("With mask")
else:
    print("Without mask")