import glob
import math
import os
import sys

from model.abstract_model import AbstractModel


class Model(AbstractModel):
    def __init__(self, detection_mode, recognition_model, tracking_method, value):
        """
        Args:
            detection_mode:
            recognition_model:
            tracking_method:
        """
        super().__init__(value)
        self.detection_model = detection_mode
        self.recognition_model = recognition_model
        self.tracking_method = tracking_method

    def predict_image(self, path_to_img: str):
        pass

    def predict_video(self, path_to_video: str):
        pass

    def print_bb_on_image(self, path_to_img: str):
        pass

    def recognize_animals_on_image(self, path_to_img: str):
        pass
