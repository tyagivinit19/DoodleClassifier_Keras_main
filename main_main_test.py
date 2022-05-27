# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 08:08:38 2020

@author: Dell
"""

import numpy as np
from tensorflow import keras

import cv2
from PIL import Image, ImageOps

new_model = keras.models.load_model('new_main_main_model')


def get_name(img_name):
    oImage = Image.open("images/" + img_name)
    size = (28, 28)
    image = ImageOps.fit(oImage, size, Image.ANTIALIAS)

    # img = image.convert('LA')
    image.save('resize.jpg')

    image = cv2.imread('resize.jpg')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    import matplotlib.pyplot as plt
    plt.imshow(gray, cmap="gray")
    plt.show()

    img = gray.reshape(28, 28, 1)

    x_test = img.reshape(1, 28, 28, 1)

    x_test = np.float32(x_test)
    y_pred = new_model.predict(x_test)

    n = np.argmax(y_pred)

    if n == 0:
        return 'Apple'

    elif n == 1:
        return 'Bucket'

    elif n == 2:
        return 'Cat'

    elif n == 3:
        return 'Clock'

    elif n == 4:
        return 'Moon'

    elif n == 5:
        return 'Rainbow'

    elif n == 6:
        return 'T.V.'

    elif n == 7:
        return 'Train'


# Image details:
# Canvas should be 280x280 px
# Background should be black
# Pencil color should be white
# Use a calligraphy white color pen for draw the image on black background
# The pen thickness should be 14px


img_name = 'test_image8.jpg'
name = get_name(img_name)

print(name)
