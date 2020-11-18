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
    left_input = tf.keras.layers.Input(input_shape)
    right_input = tf.keras.layers.Input(input_shape)

    base = tf.keras.applications.MobileNetV2(input_shape=input_shape,
                                             alpha=cfg.NN.ALPHA, weights='imagenet', include_top=False)

    for layer in base.layers:
        layer.trainable = trainable

    # conv = base.get_layer('block_16_project_BN').output
    # # Change 112 to whatever is the size of block_16_project_BN, "112" value is correct for 0.35 ALPHA, 448 is for 1.4
    # # Depends on your output complexity you might want to add another Conv2D layers (like one commented out displayed below)
    # conv = tf.keras.layers.Conv2D(240, padding="same", kernel_size=3, strides=1, activation="relu")(conv)
    # # conv = tf.keras.layers.Conv2D(240, padding="same", kernel_size=3, strides=1, use_bias=False)(conv)
    # conv = tf.keras.layers.BatchNormalization()(conv)
    # conv = tf.keras.layers.Activation('relu')(conv)
    # conv = tf.keras.layers.Dense(2048, activation='sigmoid')(conv)

    conv = tf.keras.Sequential([
        base,
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(2048, activation='sigmoid')
    ])

    encoded_l = conv(left_input)
    encoded_r = conv(right_input)
    merge_layer = tf.keras.layers.Lambda(euclidean_dist)([encoded_l, encoded_r])
    prediction = tf.keras.layers.Dense(1, activation='sigmoid')(merge_layer)

    model = tf.keras.Model(inputs=[left_input, right_input], outputs=prediction)

    # divide by 2 since d/dweight learning_rate * weight^2 = 2 * learning_rate * weight
    # see https://arxiv.org/pdf/1711.05101.pdf
    regularizer = tf.keras.regularizers.l2(cfg.TRAIN.WEIGHT_DECAY / 2)

    for weight in model.trainable_weights:
        with tf.keras.backend.name_scope('weight_regularizer'):
            model.add_loss(lambda: regularizer(weight))

    return model
