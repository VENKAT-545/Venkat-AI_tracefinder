import cv2
import numpy as np

IMG_SIZE = 128

def preprocess_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image = image.astype("float32") / 255.0
    image = np.expand_dims(image, axis=0)
    return image
