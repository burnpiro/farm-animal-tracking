import tensorflow as tf
import tensorflow.keras.backend as K

from model.siamese.config import cfg


base_models = {
    "MobileNetV2": tf.keras.applications.MobileNetV2,
    "ResNet101V2": tf.keras.applications.ResNet101V2,
    "EfficientNetB5": tf.keras.applications.EfficientNetB5,
}

default_layers_to_train = {
    "MobileNetV2": [""],
    "ResNet101V2": ["conv4"],
    "EfficientNetB5": ["block3c", "block3b"],
}

default_base_layers = {
    "MobileNetV2": "block_10_project_BN",
    "ResNet101V2": "conv4_block23_out",
    "EfficientNetB5": "block3c_add"
}


def euclidean_dist(vect):
    x, y = vect
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    result = K.maximum(sum_square, K.epsilon())
    return result


def create_model(trainable=False, base_model="MobileNetV2", layers_to_train=None, base_layer_name=None):
    if layers_to_train is None:
        layers_to_train = default_layers_to_train[base_model]
    if base_layer_name is None:
        base_layer_name = default_base_layers[base_model]

    input_shape = (cfg.NN.INPUT_SIZE, cfg.NN.INPUT_SIZE, 3)

    base_class = base_models[base_model]

    base = base_class(input_shape=input_shape, weights="imagenet", include_top=False)

    for layer in base.layers:
        layer.trainable = False
        if layers_to_train:
            if layer.name.startswith(tuple(layers_to_train)):
                layer.trainable = trainable

    x = base.get_layer(base_layer_name).output

    input = base.input
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    # x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(64, activation="linear")(x)
    x = tf.keras.layers.Lambda(lambda tensor: tf.math.l2_normalize(tensor, axis=1))(x)

    model = tf.keras.Model(inputs=input, outputs=x)

    return model
