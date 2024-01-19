import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import pandas as pd
import cv2
import numpy as np

data = pd.read_csv('fer2013.csv')

pixels = data['pixels'].tolist()
labels = data['emotion'].tolist()

pixels = [np.array(pixel.split(), dtype='uint8') for pixel in pixels]
pixels = np.array(pixels) / 255.0

pixels = pixels.reshape(-1, 48, 48, 1)
labels = tf.keras.utils.to_categorical(labels, num_classes=7)

x_train, x_val, y_train, y_val = train_test_split(pixels, labels, test_size=0.2, random_state=42)

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(7, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

datagen = ImageDataGenerator(rotation_range=10, width_shift_range=0.1, height_shift_range=0.1, zoom_range=0.1)
datagen.fit(x_train)

history = model.fit(datagen.flow(x_train, y_train, batch_size=64),
                    steps_per_epoch=len(x_train) / 64,
                    epochs=30,
                    validation_data=(x_val, y_val))

model.save('facial_expression_model.h5')
