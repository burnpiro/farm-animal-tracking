import time

import numpy as np
import os
import datetime
from absl import app, flags
from absl.flags import FLAGS

from data.data_generator import DataGenerator
from data.names import names
from helpers.score_processing import cm_analysis, classification_report_latex
from model.siamese.model_generator import create_model, base_models
from model.siamese.config import cfg
from data.siamese_evaluator import SiameseEvaluator


flags.DEFINE_string(
    "weights",
    "siam-147-0.001-block3c_add_0.2488.h5",
    "weights name",
)

flags.DEFINE_string(
    "datatype",
    "train",
    "weights name",
)

flags.DEFINE_string(
    "vectors",
    "model/siamese/vectors/vecs-conc-EfficientNetB5.tsv",
    "path to vectors tsv",
)

flags.DEFINE_string(
    "meta",
    "model/siamese/vectors/meta-conc-EfficientNetB5.tsv",
    "path to meta tsv",
)

WEIGHTS_DIR = "model/siamese/weights"

base_model = list(base_models.keys())[2]  # MobileNetV2
flags.DEFINE_string('target', './crop_images/5.jpg', 'path to input image')
flags.DEFINE_string('source', './crop_images/1.jpg', 'path to input image')


def generate_test_dir(basemodel):
    test_dir = os.path.join(
        "experiments", "siamese", basemodel
    )

    if not os.path.isdir(test_dir):
        os.mkdir(test_dir)

    out_dir = os.path.join(
        test_dir, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    )

    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    return out_dir


def main(_argv):
    out_dir = generate_test_dir(base_model)

    model = create_model(base_model=base_model)
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
    cm_analysis(conf_matrix, names, filename=os.path.join(out_dir, f"conf_matrix_{base_model}.png"))
    print(class_report)
    classification_report_latex(class_report, filename=os.path.join(out_dir, f"class_report_{base_model}.txt"))


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
