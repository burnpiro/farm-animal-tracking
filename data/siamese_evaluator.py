import glob
import math
import os
import cv2
import json
import codecs
import operator

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
from typing import List, Tuple
from tqdm import tqdm
from data.names import names
from scipy.spatial.distance import euclidean
from data.data_generator import DataGenerator
from model.siamese.siamese_model import DefaultSiameseModel

tqdm.pandas()


class SiameseEvaluator:
    def __init__(self, model: DefaultSiameseModel, dataset: List[Tuple[str, int]]):
        """

        Args:
            model: DefaultSiameseModel
            dataset: List[Tuple[str, str]] - list of pairs [image_path, class]
        """
        self.model = model
        self.dataset = dataset
        self.avg_vectors = {}

    def set_avg_vectors(self, vectors_path, meta_path):
        """

        Args:
            vectors_path: str - path to avg vectors tsv
            meta_path: str - path to avg vectors classes tsv

        Returns: None

        """
        vectors = np.loadtxt(vectors_path, delimiter='\t')
        meta = np.loadtxt(meta_path, delimiter='\t')

        for i, class_id in enumerate(meta):
            self.avg_vectors[class_id] = vectors[i]

    def compare_mean_with_vectors(self, mean):
        best_class = None
        best_distance = None
        for class_id, vector in self.avg_vectors.items():
            distance = euclidean(vector, mean)
            if best_distance is None or best_distance > distance:
                best_distance = distance
                best_class = class_id

        return best_class

    def run_evaluation(self, interval: int = 1, compare_type="groups"):
        """

        Args:
            interval: int - how many frames to take for avg evaluation
            compare_type: str - "groups" or "individual"

        Returns:
            confusion_matrix, classification_report
        """
        data_per_class = {}
        predictions = []
        for (file, class_id) in self.dataset:
            if class_id not in data_per_class:
                data_per_class[class_id] = []
            data_per_class[class_id].append(file)

        pbar = tqdm(total=len(self.dataset))
        for class_id, images in data_per_class.items():
            for i in range(0, len(images), interval):
                pbar.update(interval)
                mapped_images = np.array(list(map(DataGenerator.process_image, images[i:i+interval])))
                embeddings = self.model.predict(mapped_images)
                best_class = None

                # Compare avg embedding from interval with default vectors
                if compare_type == "groups":
                    mean_val = np.mean(embeddings, axis=0)
                    best_class = self.compare_mean_with_vectors(mean_val)

                # Get dominant value from comparison with default vectors
                if compare_type == "individual":
                    classes = list(map(self.compare_mean_with_vectors, embeddings))
                    best_class = max(set(classes), key=classes.count)

                predictions.append((class_id, best_class))

        y_true = [int(i[0]) for i in predictions]
        y_pred = [int(i[1]) for i in predictions]
        conf_matrix = confusion_matrix(y_true, y_pred)
        class_report = classification_report(y_true, y_pred, target_names=names, output_dict=True)

        return conf_matrix, class_report
