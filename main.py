import matplotlib.pyplot as plt
import tensorflow as tf

from testing import test

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))    #720 pixel grid

model.add(tf.keras.layers.Dense(256, activation='relu'))
model.add(tf.keras.layers.Dropout(0.3)) 
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dropout(0.3)) 
model.add(tf.keras.layers.Dense(10, activation='softmax'))
 

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics = ['accuracy'])

model.fit(x_train, y_train, epochs=10)

model.save('handwritten.keras')

test(x_test, y_test)