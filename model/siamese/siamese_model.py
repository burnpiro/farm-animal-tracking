from abc import ABC
import os
import tensorflow as tf
import tensorflow.keras.backend as K
from model.siamese.config import cfg
from model.siamese.model_generator import create_model


dirname = os.path.dirname(__file__)
default_weights = os.path.join(dirname, cfg.MODEL.WEIGHTS_PATH, 'siam-model-91_0.0518_0.5930.h5')


class DefaultSiameseModel(ABC):
    """
    Default siamese model. It wraps TF model and provides easier to understand API.
    """
    def __init__(self, weights_path=default_weights, trainable=False):
        super().__init__()
        self.siamese_net = create_model(trainable)
        self.siamese_net.load_weights(weights_path)

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
        siamese_predictions = self.siamese_net(boxes_tensors).numpy()

        return siamese_predictions

    @staticmethod
    def euclidean_dist(vect: tuple):
        """
        calculates euclidean distance between two points
        Args:
            vect: Tuple; (x,y) point to calculate distance between

        Returns:
            Distance
        """
        x, y = vect
        sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
        result = K.maximum(sum_square, K.epsilon())
        return result