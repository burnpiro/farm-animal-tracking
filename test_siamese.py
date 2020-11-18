import cv2
import time
import numpy as np
import tensorflow as tf
from absl import app, flags
from absl.flags import FLAGS
from siamese.model import create_model
from siamese.config import cfg

flags.DEFINE_string('weights', './siam-0.65.h5',
                    'path to weights file')
flags.DEFINE_string('target', './crop_images/5.jpg', 'path to input image')
flags.DEFINE_string('source', './crop_images/1.jpg', 'path to input image')

def main(_argv):
    model = create_model()

    # We don't have weights yet
    # model.load_weights(FLAGS.weights)

    source_image = tf.keras.preprocessing.image.load_img(FLAGS.target,
                                                       target_size=(cfg.NN.INPUT_SIZE, cfg.NN.INPUT_SIZE))

    source_image = tf.keras.preprocessing.image.img_to_array(source_image)
    source_image = np.expand_dims(source_image, axis=0)
    source_image - tf.keras.applications.mobilenet_v2.preprocess_input(source_image)

    target_image = tf.keras.preprocessing.image.load_img(FLAGS.target,
                                                       target_size=(cfg.NN.INPUT_SIZE, cfg.NN.INPUT_SIZE))

    target_image = tf.keras.preprocessing.image.img_to_array(target_image)
    target_image = np.expand_dims(target_image, axis=0)
    target_image - tf.keras.applications.mobilenet_v2.preprocess_input(target_image)

    t1 = time.time()
    pred = np.squeeze(model.predict([source_image, target_image]))
    t2 = time.time()
    processing_time = t2 - t1
    print(f'processing time: {processing_time}')
    print(pred)


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass