from model.siamese.config import cfg
import tensorflow as tf
import numpy as np
import math
from abc import ABC
import os


def create_model(trainable=True):
    input_shape = (cfg.NN.INPUT_SIZE, cfg.NN.INPUT_SIZE, 3)
    input_tensor = tf.keras.layers.Input(shape=input_shape)

    base = tf.keras.applications.MobileNetV2(input_shape=input_shape,
                                             alpha=cfg.NN.ALPHA, weights='imagenet', include_top=False)

    for layer in base.layers:
        layer.trainable = trainable

    x = input_tensor
    x = base(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(512, activation='tanh')(x)
    x = tf.keras.layers.Lambda(lambda tensor: tf.math.l2_normalize(
        tensor, axis=1), name='embedding')(x)
    x = tf.keras.layers.Dense(16, activation='linear')(x)

    model = tf.keras.Model(inputs=input_tensor, outputs=x)
    return model


def create_embedding_model(classif_model):
    return tf.keras.Model(classif_model.input, classif_model.get_layer('embedding').output)


dirname = os.path.dirname(__file__)
default_weights = os.path.join(
    dirname, cfg.MODEL.WEIGHTS_PATH, 'classif-model-6_0.2701_1.4515.h5')


class ClassificationModel(ABC):
    """
    Model based on classificator with last layer removed. It wraps TF model and provides easier to understand API.
    """

    def __init__(self, weights_path=default_weights, trainable=True):
        super().__init__()
        self.net = create_model(trainable)
        self.net.load_weights(weights_path)

        # Remove last layer
        self.net = create_embedding_model(self.net)

    def predict(self, images: list):
        """
        Calculate embeddings for each input image
        Args:
            images: List<Tensor>, list of images to process

        Returns: List
            List of embeddings
        """
        images = list(map(lambda x: tf.image.resize(x, size=(
            cfg.NN.INPUT_SIZE, cfg.NN.INPUT_SIZE)), images))

        boxes_tensors = tf.stack(images)
        boxes_tensors = tf.keras.applications.mobilenet.preprocess_input(boxes_tensors)
        predictions = self.net(boxes_tensors).numpy()

        return predictions

