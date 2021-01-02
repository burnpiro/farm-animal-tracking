import sys
import cv2
import os
import datetime
import json
import codecs
import numpy as np
from tqdm import tqdm

if len(sys.argv) != 2:
    print(f"USAGE: {sys.argv[0]} <path_to_video>")
    exit()

from model.detection_model.detection_model import DefaultDetectionModel
from model.siamese.siamese_model import DefaultSiameseModel
from model.tracker.default_tracker import DefaultTracker
from model.model import Model
from data.evaluator import Evaluator

names = [
    "James",
    "Robert",
    "William",
    "Bob",
    "Charles",
    "Anthony",
    "Paul",
    "Steven",
    "Kevin",
    "George",
    "Brian",
    "Edward",
    "Gary",
    "Eric",
    "Larry",
    "Scott",
    "Frank",
]
model = Model(DefaultDetectionModel(), DefaultSiameseModel(), DefaultTracker(names))

evaluator = Evaluator(model, ["test.mp4"], ["data/tracking/01/pigs_tracking.json"])
scores, annotations, paths = evaluator.run_evaluation_for_video(
    "test.mp4", "data/tracking/01/pigs_tracking.json", "tracking_only", 75
)
print(scores)
scores = {
    key: {"abs_err": value, "mae": value / len(paths[key])}
    for key, value in scores.items()
}

out_dir = os.path.join(
    "experiments", datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
)

if not os.path.isdir(out_dir):
    os.mkdir(out_dir)

for obj_id, annotation in annotations.items():
    cv2.imwrite(
        os.path.join(out_dir, f"{obj_id}_compare.jpg"),
        Evaluator.draw_paths_comparison(
            annotation[75 : len(paths[obj_id])], paths[obj_id]
        ),
    )

json.dump(
    annotations,
    codecs.open(os.path.join(out_dir, "annotations.json"), "w", encoding="utf-8"),
    sort_keys=False,
    separators=(",", ":"),
)
json.dump(
    paths,
    codecs.open(os.path.join(out_dir, "out.json"), "w", encoding="utf-8"),
    sort_keys=False,
    separators=(",", ":"),
)
json.dump(
    scores,
    codecs.open(os.path.join(out_dir, "scores.json"), "w", encoding="utf-8"),
    indent=2,
    sort_keys=False,
    separators=(",", ":"),
)
