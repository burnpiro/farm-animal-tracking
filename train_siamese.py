from datetime import datetime

import tensorflow as tf
import tensorflow_addons as tfa
from absl import app

from data.data_generator import DataGenerator
from model.siamese.config import cfg
from model.siamese.model_generator import create_model, base_models

TRAINABLE = False

base_model = list(base_models.keys())[0]  # MobileNetV2

target = "./crop_images/1.jpg"
source = "./crop_images/5.jpg"

WEIGHTS = "siam-model-0.55.h5"
WEIGHTS_DIR = "model/siamese/weights"


def main(_argv):
    model = create_model(trainable=TRAINABLE, base_model=base_model)
    # model.load_weights(f"weights/siamese/siam-{base_model}-91_0.0518_0.5930.h5")
    try:
        tf.keras.utils.plot_model(
            model,
            to_file=f"assets/{base_model}_model_fig.png",
            show_shapes=True,
            expand_nested=True,
        )
    except ImportError as e:
        print(f"Failed to plot keras model: {e}")

    if TRAINABLE:
        model.load_weights(WEIGHTS)

    ds_generator = DataGenerator(
        file_ext=["png", "jpg"],
        folder_path="data/filter_aug/train",
        exclude_aug=True,
        step_size=4,
    )
    test_data = DataGenerator(
        training=False,
        file_ext=["png", "jpg"],
        folder_path="data/filter_aug/test",
        step_size=4,
    )
    ds_generator.add_dataset(test_data.org_images)

    # train_ds = ds_generator.get_dataset()

    learning_rate = cfg.TRAIN.LEARNING_RATE

    # optimizer = tf.keras.optimizers.RMSprop(lr=learning_rate)
    optimizer = tf.keras.optimizers.Adam(lr=learning_rate)
    loss_fun = tfa.losses.TripletSemiHardLoss()
    model.compile(loss=loss_fun, optimizer=optimizer, metrics=[])

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        WEIGHTS_DIR + "/" + base_model + "/siam-{epoch}_{loss:.4f}_{val_loss:.4f}.h5",
        monitor="val_loss",
        verbose=1,
        save_best_only=True,
        save_weights_only=True,
        mode="min",
    )
    # stop = tf.keras.callbacks.EarlyStopping(monitor="loss", patience=cfg.TRAIN.PATIENCE, mode="min")
    # reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor="loss", factor=0.6, patience=5, min_lr=1e-6, verbose=1,
    #                                                  mode="min")

    # Define the Keras TensorBoard callback.
    logdir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)


    model.fit(
        ds_generator,
        epochs=cfg.TRAIN.EPOCHS,
        callbacks=[tensorboard_callback, checkpoint],
        verbose=1,
        validation_data=test_data,
    )


if __name__ == "__main__":
    try:
        app.run(main)
    except SystemExit:
        pass
