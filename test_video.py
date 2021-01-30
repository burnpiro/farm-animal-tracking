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

if len(sys.argv) != 2:
    print(f"USAGE: {sys.argv[0]} <path_to_video>")
    exit()


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
    sys.argv[1],
]
num_of_pigs_per_video = [16]

detection_obj = DefaultDetectionModel()
siamese_obj = DefaultSiameseModel(weights_path=weights_dir, base_model=base_model)
trackers = [
    "AvgEmbeddingTracker"
]

for idx in range(0, len(videos_paths)):
    for tracker in trackers:
        selectedTracker = None
        if tracker == "DefaultTracker":
            selectedTracker = DefaultTracker(names)
        if tracker == "AvgEmbeddingTracker":
            selectedTracker = AvgEmbeddingTracker(names, vectors_path=vectors_dir, meta_path=meta_dir, max_jump=0.2)
        if tracker == "KalmanTracker":
            selectedTracker = Tracker(num_of_pigs_per_video[idx])
        video_path = videos_paths[idx]

        model = Model(
            detection_obj,
            siamese_obj,
            selectedTracker,
        )

        model.predict_video(video_path, out_path="out_"+tracker+".mp4")
