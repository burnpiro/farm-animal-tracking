import sys
import cv2
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

for obj_id, annotation in annotations.items():
    cv2.imwrite(f"{obj_id}_compare.jpg", Evaluator.draw_paths_comparison(annotation[75:len(paths[obj_id])], paths[obj_id]))
