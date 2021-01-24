import numpy as np
import cv2

from helpers.bb_helper import pre_process_boxes, bbox_to_position
from model.tracker.default_track import DefaultTrack
from model.tracker.abstract_classes import AbstractTracker

font = cv2.FONT_HERSHEY_SIMPLEX


class DefaultTracker(AbstractTracker):
    def __init__(self, labels: list = None) -> None:
        super().__init__()
        self.tracks = None
        self.labels = labels
        self.reset_tracker()

    def reset_tracker(self) -> None:
        """
            Initializes tracker and resets tracks
        Returns: None

        """
        self.tracks = []

    def get_history(self):
        """

        Returns:
            Dict<object_id, List<(x,y)>>
            Object with list of positions for every tracking object
        """
        return {track.track_id: track.get_history() for track in self.tracks}

    def skip_empty_frame(self):
        for track_id, track in enumerate(self.tracks):
            track.update_with_prev_value()

    def get_track_class(self, track):
        return track.track_id

    def draw_tracked_objects(self, image_np: np.ndarray):
        result = image_np.copy()
        width = result.shape[1]
        height = result.shape[0]

        for track in self.tracks:
            last_history = track.get_history()[-1]
            ymin, xmin, ymax, xmax = track.history[-1]
            result = cv2.rectangle(
                result,
                (int(xmin * width), int(ymin * height)),
                (int(xmax * width), int(ymax * height)),
                (0, 255, 0),
                2,
            )
            label = str(self.get_track_class(track))
            if self.labels is not None:
                label = self.labels[int(self.get_track_class(track))-1]
            cv2.putText(
                result,
                label,
                (int(last_history[0]*width), int(last_history[1]*height)),
                font,
                0.99,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

        return result

    def run(self, boxes, embeddings, **kwargs):
        """
        Runs tracking for current list of boxes, this is based only on proximity of the previous bboxes
        Args:
            **kwargs:
            embeddings: List of embeddings for cropped bboxes
            boxes: Dict<bbox: class_name>

        Returns: None

        """
        boxes = pre_process_boxes(boxes)
        boxes = np.array(boxes)
        if boxes.shape[0] == 0:
            return None

        if len(self.tracks) < boxes.shape[0]:
            for i in range(len(self.tracks), boxes.shape[0]):
                self.tracks.append(DefaultTrack(i))

        distance_to_tracks = np.empty((boxes.shape[0], len(self.tracks)))
        for box_id, bbox in enumerate(boxes):
            for track_id, track in enumerate(self.tracks):
                distance_to_tracks[box_id][track_id] = track.get_distance_to_box(bbox)

        updated_tracks = []
        while np.min(distance_to_tracks) != np.inf:
            smallest_distance_pos = np.unravel_index(
                distance_to_tracks.argmin(), distance_to_tracks.shape
            )
            self.tracks[smallest_distance_pos[1]].update(
                boxes[smallest_distance_pos[0]], embeddings[smallest_distance_pos[0]]
            )
            updated_tracks.append(smallest_distance_pos[1])
            distance_to_tracks[smallest_distance_pos[0], :] = np.inf
            distance_to_tracks[:, smallest_distance_pos[1]] = np.inf

        if len(updated_tracks) < len(self.tracks):
            for track_id, track in enumerate(self.tracks):
                if track_id not in updated_tracks:
                    track.update([-1, -1, -1, -1], None)
