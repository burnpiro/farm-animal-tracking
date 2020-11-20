import tensorflow as tf
import tensorflow.keras.backend as K

from siamese.config import cfg


def euclidean_dist(vect):
    x, y = vect
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    result = K.maximum(sum_square, K.epsilon())
    return result


def create_model(trainable=False):
    input_shape = (cfg.NN.INPUT_SIZE, cfg.NN.INPUT_SIZE, 3)
    input_layer = tf.keras.layers.Input(input_shape)

    base = tf.keras.applications.MobileNetV2(input_shape=input_shape,
                                             alpha=cfg.NN.ALPHA, weights='imagenet', include_top=False)

    for layer in base.layers:
        layer.trainable = trainable

    conv = tf.keras.Model(
        inputs=base.input,
        outputs=tf.keras.layers.Dense(cfg.NN.DENSE_LAYER_SIZE, activation=None)(
            tf.keras.layers.Flatten()(base.get_layer('block_10_project_BN').output))
    )

    encoded = conv(input_layer)
    out = tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))(encoded)

    model = tf.keras.Model(inputs=input_layer, outputs=out)

    return model
