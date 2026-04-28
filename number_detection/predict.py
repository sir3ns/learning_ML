# stop TensorFlow from trying GPU detection.
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

def read():
	img = Image.open("data.jpg").convert("L").resize((28, 28))
	arr = np.array(img)
	arr = 255 - arr
	flat_arr = arr.flatten()
	return flat_arr


def predict_digit(pixel_array):
    model = load_model("model_1000_500.h5", compile=False)
    img = np.array(pixel_array, dtype=np.float32)

    if img.shape != (784,):
        raise ValueError("Input must be a 784-length array")

    img = img / 255.0
    img = img.reshape(1, 784)

    prediction = model.predict(img)
    digit = np.argmax(prediction)

    print(digit, prediction)


predict_digit(read())