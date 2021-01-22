import tensorflow as tf
gpu = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpu[0], True)

from model.siamese.classification_model import create_model
import numpy as np
from model.siamese.config import cfg
from absl import app
from tensorflow.keras.models import Model

TRAINABLE = False

target = './crop_images/1.jpg'
source = './crop_images/5.jpg'

WEIGHTS = './siam-model-0.55.h5'
WEIGHTS_DIR = 'model/weights'


def preprocess(x):
    x = tf.keras.applications.mobilenet.preprocess_input(x)
    return x


def main(_argv):
    model = create_model(trainable=TRAINABLE)
    model.load_weights(
        'model/weights/classif-model-1_1.6427_1.6386_acc_0.5618.h5')

    if TRAINABLE:
        model.load_weights(WEIGHTS)

    
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=preprocess
    )
    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=preprocess)
    ds_generator = train_datagen.flow_from_directory(
        'model/detection_model/images/augmented', batch_size=32, target_size=(cfg.NN.INPUT_SIZE, cfg.NN.INPUT_SIZE), class_mode='sparse')
    test_data = test_datagen.flow_from_directory('model/detection_model/images/test', batch_size=32, target_size=(
        cfg.NN.INPUT_SIZE, cfg.NN.INPUT_SIZE), shuffle=False, class_mode='sparse')


    learning_rate = cfg.TRAIN.LEARNING_RATE
    optimizer = tf.keras.optimizers.Adam(lr=learning_rate)

    loss_fun = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(loss=loss_fun, optimizer=optimizer, metrics=['accuracy'])

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        WEIGHTS_DIR+"/classif-model-{epoch}_{loss:.4f}_{val_loss:.4f}_acc_{val_accuracy:.4f}.h5", monitor="val_accuracy", verbose=1,
        save_best_only=True,
        save_weights_only=True, mode="min"
    )

    model.fit(
        ds_generator,
        epochs=200,
        callbacks=[checkpoint],
        verbose=1,
        validation_data=test_data,
        initial_epoch=1,
    )

    embeddings = []
    classes = []
    embed_model = Model(model.input, model.get_layer('embedding').output)
    batches = 0
    for x, y in test_data:
        embeddings.append(embed_model.predict(x))
        classes.append(y)
        batches += 1
        if batches >= test_data.n / test_data.batch_size:
            break

    embeddings = np.concatenate(embeddings, axis=0)
    classes = np.concatenate(classes, axis=0).astype(np.int)

    np.savetxt("vecs.tsv", embeddings, delimiter='\t')
    np.savetxt("meta.tsv", classes, delimiter='\t')


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
