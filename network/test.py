import numpy as np
import tensorflow as tf
from tensorflow import keras

import train

DEFAULT_SAVE_FOLDER = "./weights"


class Network:
    def __init__(self, path=None):
        if path is None:
            # init model from scratch and train
            self.model = train.train_model()
        self.model = keras.models.load_model(DEFAULT_SAVE_FOLDER)

    def prepare_image(self, file_path):
        # predict image for prediction
        img = tf.keras.utils.load_img(file_path, target_size=(224, 224))
        img_array = keras.preprocessing.image.img_to_array(img)
        img_array_expanded_dims = np.expand_dims(img_array, axis=0)
        return keras.applications.vgg16.preprocess_input(img_array_expanded_dims)

    def predict(self, image_path):
        input_image = self.prepare_image(image_path)
        prediction = self.model.predict(input_image)
        # take the first 10 of the predictions
        top_k_values, top_k_indices = tf.nn.top_k(prediction, k=10)
        return [top_k_values, top_k_indices]