import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv1D
from keras.layers import MaxPooling1D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import BatchNormalization
import numpy as np


data = keras.datasets.mnist                                                  #load dataset
(train_images,train_labels),(test_images,test_labels) = data.load_data()

train_images = train_images/255        #preprocessing data
test_images = test_images/255

model = Sequential()
model.add(Conv1D(56, 3, activation='relu', input_shape=(28, 28)))
model.add(MaxPooling1D(2,2))
model.add(Flatten())
model.add(Dense(784,activation='relu'))
model.add(Dense(10, activation='softmax'))

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(train_images, train_labels, epochs=5)

''' Accuracy Output Of Training Set
Epoch 1/5
60000/60000 [==============================] - 13s 210us/step - loss: 0.1662 - accuracy: 0.9496
Epoch 2/5
60000/60000 [==============================] - 12s 204us/step - loss: 0.0617 - accuracy: 0.9810
Epoch 3/5
60000/60000 [==============================] - 12s 201us/step - loss: 0.0373 - accuracy: 0.9885
Epoch 4/5
60000/60000 [==============================] - 13s 217us/step - loss: 0.0276 - accuracy: 0.9909
Epoch 5/5
60000/60000 [==============================] - 13s 218us/step - loss: 0.0201 - accuracy: 0.9933
'''

test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test Accuracy = ',test_acc)

''' Accuracy Output of test set
10000/10000 [==============================] - 0s 42us/step
Test Accuracy =  0.9842000007629395
'''

pred = model.predict(test_images)
print('Number Predicted',np.argmax(pred[725]))
print('Predicted Numbers Label',test_labels[725])

''' Output
Number Predicted 1
Predicted Numbers Label 1
'''
