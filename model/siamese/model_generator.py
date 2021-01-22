import tensorflow as tf
import tensorflow.keras.backend as K

from model.siamese.config import cfg


base_models = {
    "MobileNetV2": tf.keras.applications.MobileNetV2,
    "ResNet101V2": tf.keras.applications.ResNet101V2,
    "EfficientNetB5": tf.keras.applications.EfficientNetB5,
}


def euclidean_dist(vect):
    x, y = vect
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    result = K.maximum(sum_square, K.epsilon())
    return result


def create_model(trainable=False, base_model="MobileNetV2"):
    input_shape = (cfg.NN.INPUT_SIZE, cfg.NN.INPUT_SIZE, 3)
    input_layer = tf.keras.layers.Input(input_shape)

    base_class = base_models[base_model]

    base = base_class(input_shape=input_shape, weights="imagenet", include_top=False)

    for layer in base.layers:
        layer.trainable = trainable

    # conv = tf.keras.Model(
    #     inputs=base.input,
    #     outputs=tf.keras.layers.Dense(2048, activation=None)(tf.keras.layers.Flatten()(base.get_layer('block_10_project_BN').output))
    # )
    # x = base.get_layer('block_10_project_BN').output
    # x = base(input)
    if base_model == list(base_models.keys())[0]:
        x = base.get_layer("block_10_project_BN").output
    if base_model == list(base_models.keys())[1]:
        x = base.get_layer("block_10_project_BN").output
    if base_model == list(base_models.keys())[2]:
        x = base.get_layer("block5g_add").output
    input = base.input
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    # x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(64, activation="linear")(x)
    x = tf.keras.layers.Lambda(lambda tensor: tf.math.l2_normalize(tensor, axis=1))(x)

    model = tf.keras.Model(inputs=input, outputs=x)

    return model
