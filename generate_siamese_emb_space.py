import io

import numpy as np
import tensorflow_datasets as tfds
from absl import app, flags
from absl.flags import FLAGS

from data.data_generator import DataGenerator
from model.siamese.model_generator import create_model, base_models
from model.siamese.config import cfg

flags.DEFINE_string(
    "weights",
    "siam-118-0.0001-1.0a_0.0633.h5",
    "weights name",
)

flags.DEFINE_string(
    "datatype",
    "train",
    "weights name",
)

WEIGHTS_DIR = "model/siamese/weights"

base_model = list(base_models.keys())[0]  # MobileNetV2


def main(_argv):
    model = create_model(base_model=base_model)
    if FLAGS.datatype != "train" and FLAGS.datatype != "test":
        FLAGS.datatype = "train"

    model.load_weights(f"{WEIGHTS_DIR}/{base_model}/{FLAGS.weights}")

    ds_generator = DataGenerator(
        file_ext=["png", "jpg"],
        folder_path=f"data/filter_aug/{FLAGS.datatype}",
        exclude_aug=True,
        step_size=15,
    )

    dataset = ds_generator.get_dataset()

    results = model.predict(dataset)

    # save pure results (embedding) and create meta mapping for each row (visualization files)
    np.savetxt(f"vecs-{FLAGS.datatype}-{base_model}.tsv", results, delimiter="\t")
    out_m = io.open(f"meta-{FLAGS.datatype}-{base_model}.tsv", "w", encoding="utf-8")
    for img, labels in tfds.as_numpy(dataset):
        [out_m.write(str(x) + "\n") for x in labels]
    out_m.close()

    # merge all embeddings per class
    per_class = {}
    idx = 0
    for img, labels in tfds.as_numpy(dataset):
        for class_id in labels:
            if class_id not in per_class:
                per_class[class_id] = []
            per_class[class_id].append(results[idx])
            idx += 1

    mean_values = None
    labels = None
    # calculate average value for each class
    for class_id, values in per_class.items():
        matrix = np.array(values)
        mean_val = np.mean(matrix, axis=0)
        if mean_values is None:
            mean_values = np.array([mean_val])
        else:
            mean_values = np.concatenate((mean_values, np.array([mean_val])), axis=0)
        if labels is None:
            labels = np.array([class_id])
        else:
            labels = np.concatenate((labels, [class_id]), axis=0)

    # save avg embedding per class to be used as visualization and for further processing
    np.savetxt(f"vecs-conc-{base_model}.tsv", mean_values, delimiter="\t")
    np.savetxt(f"meta-conc-{base_model}.tsv", labels, fmt="%i", delimiter="\t")
    np.savetxt(
        f"emb_space.csv", np.concatenate((mean_values, labels), axis=1), delimiter="\t"
    )


if __name__ == "__main__":
    try:
        app.run(main)
    except:
        pass
