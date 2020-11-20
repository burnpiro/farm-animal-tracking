import time

import numpy as np
from absl import app, flags
from absl.flags import FLAGS

from data.data_generator import DataGenerator
from siamese.model import create_model
from siamese.config import cfg

flags.DEFINE_string('weights', f'{cfg.MODEL.WEIGHTS_PATH}siam-model-0.0012.h5',
                    'path to weights file')
flags.DEFINE_string('target', './crop_images/5.jpg', 'path to input image')
flags.DEFINE_string('source', './crop_images/1.jpg', 'path to input image')


def main(_argv):
    model = create_model()

    model.load_weights(FLAGS.weights)

    source_image = DataGenerator.process_image(FLAGS.source, to_input=True)
    target_image = DataGenerator.process_image(FLAGS.target, to_input=True)

    t1 = time.time()
    pred = model.predict(source_image)
    pred2 = model.predict(target_image)
    t2 = time.time()
    processing_time = t2 - t1
    distance = np.linalg.norm(pred - pred2)
    print(f'processing time: {processing_time}')
    print(f'Distance between two images is: {distance}')
    return distance


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
