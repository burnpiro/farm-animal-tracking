import numpy as np
from abc import ABC
from scipy.optimize import linear_sum_assignment

from helpers.bb_helper import pre_process_boxes
from model.tracker.track import Track
from model.tracker.abstract_classes import AbstractTracker


class Tracker(AbstractTracker):
    def __init__(self, paths_num) -> None:
        super().__init__()
        self.appearance_weight = 0.8
        self.paths_num = paths_num
        self.reset_tracker()

    def reset_tracker(self) -> None:
        """
            Initializes tracker and resets tracks
        Returns: None

        """
        # self.tracks = [Track() for _ in range(paths_num)]
        self.tracks = None

    @staticmethod
    def boxes_to_xywh(boxes):
        wh = boxes[:, 2:] - boxes[:, :2]
        xy = boxes[:, :2] + wh / 2
        return np.concatenate([xy, wh], axis=-1)

    def get_history(self):
        """

        Returns:
            Dict<object_id, List<(x,y)>>
            Object with list of positions for every tracking object
        """
        history = {track.track_id: np.array(track.get_history()) for track in self.tracks}
        # print(history)
        return history

    def similarity_matrix(self, new_bboxes, new_embeddings):
        matrix = np.empty((self.paths_num, new_embeddings.shape[0]))
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                matrix[i, j] = self.tracks[i].get_distance(
                    new_bboxes[j], new_embeddings[j]
                )
        return matrix

    def position_similarity_matrix(self, new_bboxes):
        matrix = np.empty((self.paths_num, new_bboxes.shape[0]))
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                matrix[i, j] = self.tracks[i].get_position_distance(
                    new_bboxes[j],
                )
        return matrix

    def appearance_similarity_matrix(self, new_embeddings):
        matrix = np.empty((self.paths_num, new_embeddings.shape[0]))
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                matrix[i, j] = self.tracks[i].get_similarity_distance(new_embeddings[j])
        return matrix

    def run(self, boxes, embeddings):
        boxes = pre_process_boxes(boxes)
        boxes = np.array(boxes)
        if boxes.shape[0] == 0:
            return None

        boxes = Tracker.boxes_to_xywh(boxes)

        if self.tracks is None:
            if boxes.shape[0] < self.paths_num:
                return None

            self.tracks = [
                Track(boxes[i], embeddings[i], i + 1) for i in range(self.paths_num)
            ]

        for track in self.tracks:
            track.predict()

        # t_1 = 9.4877
        t_1 = 8
        t_2 = 0.3
        position_similarity_matrix = self.position_similarity_matrix(boxes)
        position_similarity_matrix[position_similarity_matrix > t_1] = t_1

        appearance_similarity_matrix = self.appearance_similarity_matrix(embeddings)
        appearance_similarity_matrix[appearance_similarity_matrix > t_2] = t_2

        admissible = np.logical_and(
            position_similarity_matrix < t_1, appearance_similarity_matrix < t_2
        )
        similarity_matrix = (
            (1 - self.appearance_weight) * position_similarity_matrix
            + self.appearance_weight * appearance_similarity_matrix
        )

        rows, cols = linear_sum_assignment(similarity_matrix)

        for i, j in zip(rows, cols):
            if not admissible[i, j]:
                continue
            # self.tracks[i].embedding = embeddings[j]
            self.tracks[i].embeddings.append(embeddings[j])
            # self.tracks[i].set_bbox(boxes[j])
            self.tracks[i].update(boxes[j])
