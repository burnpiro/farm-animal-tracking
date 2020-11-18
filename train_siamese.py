import tensorflow as tf
from datetime import datetime
from siamese.model import create_model
from siamese.config import cfg
import numpy as np

TRAINABLE = False

target = './crop_images/1.jpg'
source = './crop_images/5.jpg'


def main():
    model = create_model(trainable=TRAINABLE)

    # if TRAINABLE:
    #     model.load_weights(WEIGHTS)

    learning_rate = cfg.TRAIN.LEARNING_RATE
    if TRAINABLE:
        learning_rate /= 10

    optimizer = tf.keras.optimizers.SGD(lr=learning_rate, decay=cfg.TRAIN.LR_DECAY, momentum=0.9, nesterov=False)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=[])

    checkpoint = tf.keras.callbacks.ModelCheckpoint("siam-model-{val_iou:.2f}.h5", monitor="val_iou", verbose=1,
                                                    save_best_only=True,
                                                    save_weights_only=True, mode="max")
    stop = tf.keras.callbacks.EarlyStopping(monitor="val_iou", patience=cfg.TRAIN.PATIENCE, mode="max")
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_iou", factor=0.6, patience=5, min_lr=1e-6, verbose=1,
                                                     mode="max")

    # Define the Keras TensorBoard callback.
    logdir="logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

    source_image = tf.keras.preprocessing.image.load_img(target,
                                                       target_size=(cfg.NN.INPUT_SIZE, cfg.NN.INPUT_SIZE))

    source_image = tf.keras.preprocessing.image.img_to_array(source_image)
    source_image = np.expand_dims(source_image, axis=0)
    source_image - tf.keras.applications.mobilenet_v2.preprocess_input(source_image)

    target_image = tf.keras.preprocessing.image.load_img(target,
                                                       target_size=(cfg.NN.INPUT_SIZE, cfg.NN.INPUT_SIZE))

    target_image = tf.keras.preprocessing.image.img_to_array(target_image)
    target_image = np.expand_dims(target_image, axis=0)
    target_image - tf.keras.applications.mobilenet_v2.preprocess_input(target_image)

    print(source_image.shape)
    print(target_image.shape)
    print(model.predict([source_image, target_image]))

    print(model.summary())


    # model.fit_generator(generator=train_datagen,
    #                     epochs=cfg.TRAIN.EPOCHS,
    #                     callbacks=[tensorboard_callback, checkpoint, reduce_lr, stop],
    #                     shuffle=True,
    #                     verbose=1)


if __name__ == "__main__":
    main()