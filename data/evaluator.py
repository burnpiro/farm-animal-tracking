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

    @staticmethod
    def compare_path_parts(
        annotations, paths, interval=10, video_frame_offset=0, eval_type="tracking_only"
    ):
        """
        Sums total distance between positions in each frame
        Args:
            annotations: Dict[class_id, List<x,y,frame_id>]
            paths: Dict[class_id, List<x,y>]
            interval: int - number of frames to be used as a part of a comparison
            video_frame_offset: int - offset for video
            eval_type: string - "all" or "tracking_only" (should include recognition in tracking)

        Returns: Dict[class_id, {
            "intervals": int,
            "parts": List<float>
        }]
            Errors per interval, per class
        """

        first_ann = list(annotations.values())[0]
        scores = {}
        for i in range(0, len(first_ann), interval):
            if i >= len(list(paths.values())[0]):
                break
            mapped_paths_with_offset = Evaluator.map_paths_to_closest(
                paths, annotations, compare_specific_position=i
            )
            for object_id, ann_path in annotations.items():
                if object_id not in scores:
                    scores[object_id] = {"interval": interval, "parts": []}
                sum_dist = 0.0
                last_num = 0
                for frame_id in range(i, i + interval):
                    if frame_id >= len(
                        mapped_paths_with_offset[object_id]
                    ) or video_frame_offset + frame_id >= len(ann_path):
                        break
                    last_num += 1
                    sum_dist += euclidean(
                        ann_path[video_frame_offset + frame_id][:2],
                        mapped_paths_with_offset[object_id][frame_id],
                    )

                if last_num == 0:
                    continue
                scores[object_id]["parts"].append(sum_dist / last_num)
        return scores

    @staticmethod
    def map_paths_to_closest(paths, annotations, compare_specific_position=0):
        """

        Args:
            paths: List[x,y] - list of x,y position
            annotations: List[x,y,c] - list of x,y position and class id
            compare_specific_position: int - compare paths in given position

        Returns:

        """
        path_mapping = {}
        assigned_paths = []
        for annotation_id, ann_path in annotations.items():
            distances = {}

            for path_id, path in paths.items():
                try:
                    distances[path_id] = euclidean(
                        path[compare_specific_position],
                        ann_path[compare_specific_position][:2],
                    )
                except:
                    # print(path_id, compare_specific_position, path)
                    print("there was an error", path_id)

            loop = 1
            # print(min(distances.items(), key=operator.itemgetter(1)))
            closest_path_idx = min(distances.items(), key=operator.itemgetter(1))[0]

            distances = list(distances.items())
            distances.sort(key=operator.itemgetter(1))

            while closest_path_idx in assigned_paths:
                try:
                    closest_path_idx = distances[loop][0]
                except:
                    print("error in loop", loop)
                    break
                loop += 1

            # print(distances, annotation_id, closest_path_idx)
            path_mapping[annotation_id] = closest_path_idx
            assigned_paths.append(closest_path_idx)

        # print(assigned_paths, path_mapping, compare_specific_position)
        new_paths = {}
        for ann_id, track_id in path_mapping.items():
            new_paths[ann_id] = paths[track_id]

        return new_paths

    def run_evaluation_for_video(
        self,
        path_to_video: str,
        path_to_annotations: str,
        eval_type: str = "all",
        video_frame_offset: int = 0,
        compare_parts: bool = False,
        compare_part_interval: int = 10,
        video_out_path=None,
    ):
        """

        Args:
            path_to_video: str - path to video file
            path_to_annotations: str - path to annotation json file
            eval_type: "all" or "tracking_only" (should include recognition in tracking)
            video_frame_offset: int - if video has offset to annotations (starts from nth frame) then enter that offset
            compare_parts: bool - should also compare paths by intervals
            compare_part_interval: int - if @compare_parts is True then this a interval for path splitting
            video_out_path: str - if you want to save the video put the path in here

        Returns: (score, annotations, paths)
            scores: Dict<object_id, {
                "total": float,
                "intervals": {
                    "interval": int,
                    "parts": List<float>
                }
            }>
            annotations: Dict<object_id, List<x,y,frame_id>
            paths: Dict<object_id, List<x,y>>
        """
        annotations = json.load(codecs.open(path_to_annotations))
        # for object_id, annotation in annotations.items():
        #     print(annotation[video_frame_offset:video_frame_offset+1])
        paths = self.model.predict_video(path_to_video, out_path=video_out_path)

        if eval_type == "tracking_only":
            paths = Evaluator.map_paths_to_closest(
                paths, annotations, compare_specific_position=0
            )

        scores = {}
        for object_id, annotation in annotations.items():
            scores[object_id] = {}

            scores[object_id]["total"] = Evaluator.compare_paths(
                annotation[video_frame_offset:], paths[object_id]
            )

        if compare_parts:
            interval_scores = Evaluator.compare_path_parts(
                annotations,
                paths,
                compare_part_interval,
                video_frame_offset=video_frame_offset,
                eval_type=eval_type,
            )
            for object_id, interval_score in interval_scores.items():
                scores[object_id]["intervals"] = interval_score

        return scores, annotations, paths

    @staticmethod
    def draw_paths_comparison(
        paths1: List[Tuple[float, float]],
        paths2: List[Tuple[float, float]],
        height: int = 450,
        width: int = 800,
        color1=(0, 255, 0),
        color2=(0, 0, 255),
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

            cv2.line(blank_image, prev_path1_point, curr_path1_point, color1, 3)
            cv2.line(blank_image, prev_path2_point, curr_path2_point, color2, 3)

        return blank_image

    @staticmethod
    def draw_path_parts_comparison(
        paths1: List[Tuple[float, float]],
        paths2: List[Tuple[float, float]],
        scores: List[float],
        interval: int,
        height: int = 450,
        width: int = 800,
        threshold={
            0: (76, 179, 105),
            0.025: (52, 179, 172),
            0.05: (51, 163, 250),
            0.1: (21, 142, 255),
            0.2: (0, 0, 255),
            100: (0, 0, 255),
        },
    ):
        blank_image = np.zeros((height, width, 3), np.uint8)

        for idx in range(1, len(paths1) - 1):
            if int(idx / interval) >= len(scores):
                break
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

            score = scores[int(idx / interval)]
            color = threshold[100]
            keys = list(threshold.keys())
            for t_id in range(len(keys)):
                if keys[t_id] <= score < keys[t_id + 1]:
                    color = list(threshold.items())[t_id][1]

            cv2.line(blank_image, prev_path2_point, curr_path2_point, color, 3)

        return blank_image
