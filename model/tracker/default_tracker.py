import numpy as np

from helpers.bb_helper import pre_process_boxes
from model.tracker.default_track import DefaultTrack
from model.tracker.abstract_classes import AbstractTracker


class DefaultTracker(AbstractTracker):
    def __init__(self, paths_num) -> None:
        super().__init__()
        self.paths_num = paths_num
        self.tracks = None
        self.initialize_tracker()

    def initialize_tracker(self) -> None:
        """
            Initializes tracker and resets tracks
        Returns: None

        """
        self.tracks = [
                DefaultTrack(i + 1) for i in range(self.paths_num)
            ]

    def get_history(self):
        """

        Returns:
            Dict<object_id, List<(x,y)>>
            Object with list of positions for every tracking object
        """
        return {track.track_id: np.array(track.get_history()) for track in self.tracks}

    def run(self, boxes, embeddings):
        """
        Runs tracking for current list of boxes, this is based only on proximity of the previous bboxes
        Args:
            embeddings: List of embeddings for cropped bboxes
            boxes: Dict<bbox: class_name>

        Returns: None

        """
        boxes = pre_process_boxes(boxes)
        boxes = np.array(boxes)
        if boxes.shape[0] == 0:
            return None

        if self.tracks is None:
            if boxes.shape[0] < self.paths_num:
                return None

            self.tracks = [
                DefaultTrack(i + 1) for i in range(self.paths_num)
            ]


