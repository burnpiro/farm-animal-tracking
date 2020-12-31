import sys
import cv2
from tqdm import tqdm

if len(sys.argv) != 2:
    print(f"USAGE: {sys.argv[0]} <path_to_video>")
    exit()

from model.detection_model.detection_model import DefaultDetectionModel
from model.siamese.siamese_model import DefaultSiameseModel
from model.tracker.tracker import Tracker
from model.model import Model

model = Model(DefaultDetectionModel(), DefaultSiameseModel(), Tracker(7))

paths = model.predict_video(sys.argv[1])

print(paths)