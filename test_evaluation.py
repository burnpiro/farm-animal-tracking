import sys
import cv2
import os
import datetime
import json
import codecs
import numpy as np
from tqdm import tqdm
import tensorflow as tf
gpu = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpu[0], True)

from model.siamese.model_generator import base_models
from model.detection_model.detection_model import DefaultDetectionModel
from model.siamese.siamese_model import DefaultSiameseModel
from model.siamese.classification_model import ClassificationModel
from model.tracker.default_tracker import DefaultTracker
from model.tracker.simple_siamese_tracker import SimpleSiameseTracker
from model.tracker.tracker import Tracker
from model.tracker.avg_embedding_tracker import AvgEmbeddingTracker
from model.tracker.default_tracker_with_path_correction import (
    DefaultTrackerWithPathCorrection,
)
from model.model import Model
from data.evaluator import Evaluator
from data.names import names
from helpers.score_processing import extract_scores, print_path_comparison


def generate_test_dir(basemodel, tracker, video):
    test_dir = os.path.join(
        "experiments", "tracking", basemodel
    )

    if not os.path.isdir(test_dir):
        os.mkdir(test_dir)

    test_tracker_dir = os.path.join(
        test_dir, tracker
    )

    if not os.path.isdir(test_tracker_dir):
        os.mkdir(test_tracker_dir)

    out = os.path.join(
        test_tracker_dir, video+"_"+datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    )

    if not os.path.isdir(out):
        os.mkdir(out)

    return out


dirname = os.path.dirname(__file__)
weights_dir = os.path.join(
    dirname, "model/siamese/weights", "MobileNetV2", "siam-118-0.0001-1.0a_0.0633.h5"
)
vectors_dir = os.path.join(
    dirname, "model/siamese/vectors", "vecs-conc-MobileNetV2.tsv"
)
meta_dir = os.path.join(
    dirname, "model/siamese/vectors", "meta-conc-MobileNetV2.tsv"
)
base_model = list(base_models.keys())[0]  # MobileNetV2

# model = Model(DefaultDetectionModel(), DefaultSiameseModel(), DefaultTracker(names))
# model = Model(DefaultDetectionModel(), DefaultSiameseModel(), Tracker(7))


videos_paths = [
    "11_nursery_high_activity_day-cropped.mp4",
    "12_nursery_low_activity_day-cropped.mp4",
    "13_nursery_low_activity_night-cropped.mp4",
    "14_nursery_medium_activity_day-cropped.mp4",
    "15_nursery_medium_activity_night-cropped.mp4"
]
annotations_paths = [
    "data/tracking/11/pigs_tracking.json",
    "data/tracking/12/pigs_tracking.json",
    "data/tracking/13/pigs_tracking.json",
    "data/tracking/14/pigs_tracking.json",
    "data/tracking/15/pigs_tracking.json",
]
start_times = [6000, 6000, 6000, 6000, 6000]
num_of_pigs_per_video = [16, 15, 16, 16, 16]

detection_obj = DefaultDetectionModel()
siamese_obj = DefaultSiameseModel(weights_path=weights_dir, base_model=base_model)
trackers = [
    "DefaultTracker",
    "AvgEmbeddingTracker"
    "KalmanTracker"
]

for idx in range(0, len(videos_paths)):
    for tracker in trackers:
        selectedTracker = None
        if tracker == "DefaultTracker":
            selectedTracker = DefaultTrackerWithPathCorrection(names)
        if tracker == "AvgEmbeddingTracker":
            selectedTracker = AvgEmbeddingTracker(names, vectors_path=vectors_dir, meta_path=meta_dir)
        if tracker == "KalmanTracker":
            selectedTracker = Tracker(num_of_pigs_per_video[idx])
        video_path = videos_paths[idx]
        annotation_path = annotations_paths[idx]
        offset_val = start_times[idx]

        model = Model(
            detection_obj,
            siamese_obj,
            selectedTracker,
        )

        evaluator = Evaluator(
            model,
            videos_paths,
            annotations_paths,
        )
        scores, annotations, paths = evaluator.run_evaluation_for_video(
            video_path,
            annotation_path,
            "tracking_only",
            video_frame_offset=offset_val,
            compare_parts=True,
            compare_part_interval=5,
            video_out_path=None
        )
        scores = extract_scores(scores, paths)

        out_dir = generate_test_dir("MobileNetV2", tracker, video_path)

        if not os.path.isdir(out_dir):
            os.mkdir(out_dir)

        for obj_id, annotation in annotations.items():
            print_path_comparison(
                out_dir,
                annotation[offset_val: offset_val+len(paths[obj_id])],
                paths[obj_id],
                obj_id,
                interval=scores[obj_id]["intervals"]["interval"],
                parts=scores[obj_id]["intervals"]["parts"],
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
