import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os

def test(x_test, y_test):
    model = tf.keras.models.load_model('handwritten.keras')

    loss, accuracy = model.evaluate(x_test, y_test)

    print(loss)
    print(accuracy)

    digit_number = 1
    version_number = 1

    while os.path.isfile(f"samples/digit{digit_number}.{version_number}.png"):
        try:

            img = cv2.imread(f"samples/digit{digit_number}.{version_number}.png")[:,:,0]
            
            img = np.invert(np.array([img]))
            img = img / 255.0  # Normalize the image
            
            prediction = model.predict(img)
            print(f"This digit is probably a {np.argmax(prediction)}")
            
            plt.imshow(img[0], cmap=plt.cm.binary)
            plt.show()
            
            version_number += 1
            
            if not os.path.isfile(f"samples/digit{digit_number}.{version_number}.png"):
                digit_number += 1
                version_number = 1  # Reset version for the next digit
        except:
            print("Error!")

