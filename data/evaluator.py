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
from typing import List, Tuple
from tqdm import tqdm
from scipy.spatial.distance import euclidean
from model.model import AbstractModel

tqdm.pandas()


class Evaluator:
    def __init__(self, model: AbstractModel, videos: List[str], annotations: List[str]):
        """

        Args:
            model: AbstractModel
            videos: List[str] - list of video paths to process
            annotations: List[str] - list of annotations for "videos" has to be the same size
        """
        if len(videos) != len(annotations):
            raise Exception("Number of videos has to be the same as num of annotations")

        self.model = model
        self.videos = videos
        self.annotations = annotations

    @staticmethod
    def compare_paths(annotation, path):
        """
        Sums total distance between positions in each frame
        Args:
            annotation: List<x,y,frame_id>
            path: List<x,y>

        Returns: float
            Total distance (Absolute Error)
        """
        sum_dist = 0.0
        for frame_id, ann_point in enumerate(annotation):
            if frame_id >= len(path):
                break
            sum_dist += euclidean(ann_point[:2], path[frame_id])

        return sum_dist

    def run_evaluation_for_video(
        self,
        path_to_video: str,
        path_to_annotations: str,
        eval_type: str = "all",
        video_frame_offset: int = 0,
    ):
        """

        Args:
            path_to_video: str - path to video file
            path_to_annotations: str - path to annotation json file
            eval_type: "all" or "tracking_only" (should include recognition in tracking)
            video_frame_offset: int - if video has offset to annotations (starts from nth frame) then enter that offset

        Returns: (score, annotations, paths)
            scores: Dict<object_id, error>
            annotations: Dict<object_id, List<x,y,frame_id>
            paths: Dict<object_id, List<x,y>>
        """
        annotations = json.load(codecs.open(path_to_annotations))
        print(len(annotations["4"]))
        paths = self.model.predict_video(path_to_video)

        if eval_type == "tracking_only":
            path_mapping = {}
            assigned_paths = []
            for annotation_id, ann_path in annotations.items():
                distances = {}
                for path_id, path in paths.items():
                    distances[path_id] = euclidean(path[0], ann_path[0][:2])

                loop = 1
                closest_path_idx = min(distances.items(), key=operator.itemgetter(1))[0]
                while closest_path_idx in assigned_paths:
                    closest_path_idx = min(
                        distances.items(), key=operator.itemgetter(1)
                    )[loop]
                    loop += 1

                path_mapping[annotation_id] = closest_path_idx
                assigned_paths.append(closest_path_idx)

            print(path_mapping)
            print(assigned_paths)
            new_paths = {}
            for ann_id, track_id in path_mapping.items():
                new_paths[ann_id] = paths[track_id]
            paths = new_paths
            del new_paths

        scores = {}
        for object_id, annotation in annotations.items():
            scores[object_id] = Evaluator.compare_paths(
                annotation[video_frame_offset:], paths[object_id]
            )

        return scores, annotations, paths

    @staticmethod
    def draw_paths_comparison(
        paths1: List[Tuple[float, float]],
        paths2: List[Tuple[float, float]],
        height: int = 450,
        width: int = 800,
    ):
        blank_image = np.zeros((height, width, 3), np.uint8)

        for idx in range(1, len(paths1) - 1):
            prev_path1_point = (
                int(paths1[idx - 1][0] * width),
                int(paths1[idx - 1][1] * height),
            )
            curr_path1_point = (
                int(paths1[idx][0] * width),
                int(paths1[idx][1] * height),
            )
            prev_path2_point = (
                int(paths2[idx - 1][0] * width),
                int(paths2[idx - 1][1] * height),
            )
            curr_path2_point = (
                int(paths2[idx][0] * width),
                int(paths2[idx][1] * height),
            )

            cv2.line(
                blank_image, prev_path1_point, curr_path1_point, (0, 255, 0), 3
            )
            cv2.line(
                blank_image, prev_path2_point, curr_path2_point, (0, 0, 255), 3
            )

        return blank_image
