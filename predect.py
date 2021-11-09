import time
seconds0 = time.time()
from tensorflow import keras
import tensorflow as tf
import tensorflow_hub as hub
seconds1 = time.time()
new_model = tf.keras.models.load_model('models/skin.h5',custom_objects={'KerasLayer':hub.KerasLayer})
# new_model.summary()
seconds2 = time.time()

img = tf.io.read_file("download.jpeg")


def decode_img(img):
    # convert the compressed string to a 3D uint8 tensor
    img = tf.image.decode_jpeg(img, channels=3)
    # Use `convert_image_dtype` to convert to floats in the [0,1] range.
    img = tf.image.convert_image_dtype(img, tf.float32)
    # resize the image to the desired size.
    return tf.image.resize(img, [299, 299])

seconds3 = time.time()
img = decode_img(img)
import numpy as np
X_test = np.zeros((1, 299, 299, 3))
X_test[0]=img
# print (X_test)

seconds4 = time.time()

def get_predictions(threshold=None):
    """
    Returns predictions for binary classification given `threshold`
    For instance, if threshold is 0.3, then it'll output 1 (malignant) for that sample if
    the probability of 1 is 30% or more (instead of 50%)
    """
    y_pred = new_model.predict(X_test)
    if not threshold:
        threshold = 0.5
    result = np.zeros((1,))
        # test melanoma probability
    if y_pred[0][0] >= threshold:
        result[0] = 1
        print("malignant")
    else:
        # else, it's 0 (benign)
        print("begnin")

y_pred = get_predictions(5)
seconds5 = time.time()
print(seconds1-seconds0)
print(seconds2-seconds1)
print(seconds3-seconds2)
print(seconds4-seconds3)
print(seconds4-seconds4)