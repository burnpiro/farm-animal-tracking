from model.siamese.config import cfg
import tensorflow as tf
import numpy as np
import math
# tf.keras.backend.set_floatx('float16')
# tf.keras.backend.set_epsilon(1e-4)

class ClassificationLayer(tf.keras.layers.Layer):
    def __init__(self, units, input_dim=32):
        super().__init__()
        w_init = tf.random_normal_initializer()
        self.w = tf.Variable(
            initial_value=w_init(shape=(input_dim, units), dtype="float32"),
            trainable=True,
        )

    def call(self, inputs):
        w = tf.math.l2_normalize(self.w, axis=1)
        return tf.matmul(inputs, w)

gaussian = np.empty((cfg.NN.INPUT_SIZE, cfg.NN.INPUT_SIZE))
sigma = cfg.NN.INPUT_SIZE/3
for i in range(gaussian.shape[0]):
    for j in range(gaussian.shape[1]):
        x = i-(gaussian.shape[0]-1)/2
        y = j-(gaussian.shape[1]-1)/2
        gaussian[i, j] = (1/(2*math.pi*sigma**2)) * \
            math.exp(-(x**2+y**2)/(2*sigma**2))

gaussian = gaussian[..., np.newaxis] * (2*math.pi*sigma**2)
gaussian = tf.constant(gaussian, dtype=tf.float32)

def create_model(trainable=False):
    input_shape = (cfg.NN.INPUT_SIZE, cfg.NN.INPUT_SIZE, 3)
    input_tensor = tf.keras.layers.Input(shape=input_shape)

    # base = tf.keras.applications.ResNet101V2(input_shape=input_shape, weights='imagenet', include_top=False)

    base = tf.keras.applications.MobileNetV2(input_shape=input_shape,
                                             alpha=cfg.NN.ALPHA, weights='imagenet', include_top=False)

    # for layer in base.layers:
    #     layer.trainable = trainable


    # x = input_tensor * gaussian
    x = input_tensor
    x = base(x)
    x = tf.keras.layers.Flatten()(x)
    # x = tf.keras.layers.Dropout(0.5)(x)
    # x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(512, activation='tanh')(x)
    x = tf.keras.layers.Lambda(lambda tensor: tf.math.l2_normalize(tensor, axis=1), name='embedding')(x)
    # x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(16, activation='linear')(x)
    # x = ClassificationLayer(16, 512)(x)
    # x = tf.keras.layers.Lambda(lambda tensor: tf.math.l2_normalize(tensor, axis=1))(x)

    model = tf.keras.Model(inputs=input_tensor, outputs=x)
    return model

def create_embedding_model(classif_model):
    return tf.keras.Model(classif_model.input, classif_model.get_layer('embedding').output)