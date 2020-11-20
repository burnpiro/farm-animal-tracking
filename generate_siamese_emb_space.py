import io

import numpy as np
import tensorflow_datasets as tfds
from absl import app, flags
from absl.flags import FLAGS

from data.data_generator import DataGenerator
from siamese.model import create_model

flags.DEFINE_string('weights', './siam-model-0.0012.h5',
                    'path to weights file')
flags.DEFINE_string('target', './crop_images/5.jpg', 'path to input image')
flags.DEFINE_string('source', './crop_images/1.jpg', 'path to input image')


def main(_argv):
    model = create_model()

    model.load_weights(FLAGS.weights)

    ds_generator = DataGenerator(training=False)

    dataset = ds_generator.get_dataset()

    results = model.predict(dataset)
    np.savetxt("vecs.tsv", results, delimiter='\t')
    out_m = io.open('meta.tsv', 'w', encoding='utf-8')
    for img, labels in tfds.as_numpy(dataset):
        [out_m.write(str(x) + "\n") for x in labels]
    out_m.close()


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
