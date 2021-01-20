import time

import numpy as np
from absl import app, flags
from absl.flags import FLAGS

from data.data_generator import DataGenerator
from model.siamese.model_generator import create_model, base_models
from model.siamese.config import cfg
from data.siamese_evaluator import SiameseEvaluator


flags.DEFINE_string(
    "weights",
    "siam-61_0.0633_0.0411.h5",
    "weights name",
)

flags.DEFINE_string(
    "datatype",
    "train",
    "weights name",
)

flags.DEFINE_string(
    "vectors",
    "model/siamese/vectors/vecs-conc-MobileNetV2.tsv",
    "path to vectors tsv",
)

flags.DEFINE_string(
    "meta",
    "model/siamese/vectors/meta-conc-MobileNetV2.tsv",
    "path to meta tsv",
)

WEIGHTS_DIR = "model/siamese/weights"

base_model = list(base_models.keys())[0]  # MobileNetV2
flags.DEFINE_string('target', './crop_images/5.jpg', 'path to input image')
flags.DEFINE_string('source', './crop_images/1.jpg', 'path to input image')


def main(_argv):
    model = create_model()
    model.load_weights(f"{WEIGHTS_DIR}/{base_model}/{FLAGS.weights}")
    ds_generator = DataGenerator(
        file_ext=["png", "jpg"],
        folder_path=f"data/filter_aug/test",
        exclude_aug=True,
        step_size=1,
    )

    images = ds_generator.__getitem__(1)
    print(images[0].shape)
    print(images[1])
    evaluator = SiameseEvaluator(model=model, dataset=ds_generator.org_images)
    evaluator.set_avg_vectors(FLAGS.vectors, FLAGS.meta)
    conf_matrix, class_report = evaluator.run_evaluation(compare_type="individual")
    print(conf_matrix)
    print(class_report)


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
