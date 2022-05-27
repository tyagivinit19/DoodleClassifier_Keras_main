# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 00:00:08 2020

@author: Dell
"""

import numpy as np

import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Activation
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import BatchNormalization


apple_x = np.load("data/apple.npy")
bucket_x = np.load("data/bucket.npy")
cat_x = np.load("data/cat.npy")
clock_x = np.load("data/clock.npy")
moon_x = np.load("data/moon.npy")
rainbow_x = np.load("data/rainbow.npy")
tv_x = np.load("data/television.npy")
train_x = np.load("data/train.npy")

print("Load Done")

apple_y = np.full((apple_x.shape[0], 1), 0, dtype=int)
bucket_y = np.full((bucket_x.shape[0], 1), 1, dtype=int)
cat_y = np.full((cat_x.shape[0], 1), 2, dtype=int)
clock_y = np.full((clock_x.shape[0], 1), 3, dtype=int)
moon_y = np.full((moon_x.shape[0], 1), 4, dtype=int)
rainbow_y = np.full((rainbow_x.shape[0], 1), 5, dtype=int)
tv_y = np.full((tv_x.shape[0], 1), 6, dtype=int)
train_y = np.full((train_x.shape[0], 1), 7, dtype=int)

print("New column")

apple_x = np.append(apple_x, apple_y, axis=1)
bucket_x = np.append(bucket_x, bucket_y, axis=1)
cat_x = np.append(cat_x, cat_y, axis=1)
clock_x = np.append(clock_x, clock_y, axis=1)
moon_x = np.append(moon_x, moon_y, axis=1)
rainbow_x = np.append(rainbow_x, rainbow_y, axis=1)
tv_x = np.append(tv_x, tv_y, axis=1)
train_x = np.append(train_x, train_y, axis=1)

print("column append")

data = np.concatenate((apple_x, bucket_x))
data = np.concatenate((data, cat_x))
data = np.concatenate((data, clock_x))
data = np.concatenate((data, moon_x))
data = np.concatenate((data, rainbow_x))
data = np.concatenate((data, tv_x))
data = np.concatenate((data, train_x))

print("concatenate all data")

np.random.shuffle(data)
data = data[:1000000, :]

print("Trim the data by 10,000,000 rows.")

x_train = data[:800000, :784]
x_test = data[800000:, :784]

y_train = data[:800000, 784].reshape(800000, 1)
y_test = data[800000:, 784].reshape(200000, 1)

print("train-test splitting")

temp_y = np.full((y_train.shape[0], 8), 0, dtype=int)

# OnehotEncoding//
for i in range(y_train.shape[0]):
    temp_y[i, y_train[i, 0]] = 1

y_train = temp_y

print("one-hot-encoding done.")

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

print("reshape for training")

print("Staring CNN..")

# CNN

model = Sequential()

model.add(Conv2D(input_shape=(28, 28, 1), filters=32, kernel_size=(3, 3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D())

model.add(Conv2D(filters=64, kernel_size=(3, 3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D())

model.add(Conv2D(filters=128, kernel_size=(3, 3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D())

model.add(Flatten())

model.add(Dense(units=64, activation='relu'))
model.add(Dense(8, kernel_initializer='uniform', activation='softmax'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

print(model.summary())

r = model.fit(x_train, y_train, batch_size=10, epochs=10)

print("Training Finished")

# Save the model

# model.save('new_main_main_model_n')    # Uncomment these two lines for save new model
# print('Model Saved..')

# Predictions

print("predictions starts")

y_pred = model.predict(x_test)

acc = 0

for i in range(y_pred.shape[0]):
    if np.argmax(y_pred[i]) == y_test[i]:
        acc = acc + 1

acc = (acc / y_test.shape[0]) * 100

print('Test Accuracy: ', acc)

plt.plot(r.history['accuracy'], label='acc')
plt.legend()
plt.show()

# a = cat_x[11, :784]
# a = a.reshape(28, 28)

# import matplotlib.pyplot as plt
# plt.imshow(a, cmap="gray")
# plt.show()
