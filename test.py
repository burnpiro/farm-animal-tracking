import sys
import cv2

if len(sys.argv) != 2:
    print(f"USAGE: {sys.argv[0]} <path_to_image>")
    exit()

from model.detection_model.detection_model import DefaultDetectionModel
from model.siamese.siamese_model import DefaultSiameseModel
from model.model import Model

model = Model(DefaultDetectionModel(), DefaultSiameseModel(), 1)

image = model.print_bb_on_image(sys.argv[1])

cv2.imwrite("result.jpg", image)
