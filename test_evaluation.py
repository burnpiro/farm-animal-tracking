import sys
import cv2
import os
import datetime
import json
import codecs
import numpy as np
from tqdm import tqdm

# if len(sys.argv) != 2:
#     print(f"USAGE: {sys.argv[0]} <path_to_video>")
#     exit()

import tensorflow as tf
gpu = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpu[0], True)

from model.detection_model.detection_model import DefaultDetectionModel
from model.siamese.siamese_model import DefaultSiameseModel
from model.tracker.default_tracker import DefaultTracker
from model.tracker.simple_siamese_tracker import SimpleSiameseTracker
from model.tracker.tracker import Tracker
from model.model import Model
from data.evaluator import Evaluator
from helpers.score_processing import extract_scores, print_path_comparison

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
# model = Model(DefaultDetectionModel(), DefaultSiameseModel(), Tracker(7))

evaluator = Evaluator(model, ["PigTrackingDataset2020/videos/01_early_finisher_high_activity_day.mp4"], [
                      "data/tracking/01/pigs_tracking.json"])

scores, annotations, paths = evaluator.run_evaluation_for_video(
    "PigTrackingDataset2020/videos/01_early_finisher_high_activity_day.mp4",
    "data/tracking/01/pigs_tracking.json",
    "tracking_only",
    0,
    compare_parts=True,
)
scores = extract_scores(scores, paths)

out_dir = os.path.join(
    "experiments", datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
)

if not os.path.isdir(out_dir):
    os.mkdir(out_dir)

for obj_id, annotation in annotations.items():
    print_path_comparison(
        out_dir,
        annotation[75: 75 + len(paths[obj_id])],
        paths[obj_id],
        obj_id,
        interval=scores[obj_id]["intervals"]["interval"],
        parts=scores[obj_id]["intervals"]["parts"],
    )

json.dump(
    annotations,
    codecs.open(os.path.join(out_dir, "annotations.json"),
                "w", encoding="utf-8"),
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
