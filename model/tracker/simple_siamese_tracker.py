import numpy as np
import cv2

from helpers.bb_helper import pre_process_boxes, bbox_to_position
from model.tracker.siamese_track import SiameseTrack
from model.tracker.default_tracker import DefaultTracker

font = cv2.FONT_HERSHEY_SIMPLEX


class SimpleSiameseTracker(DefaultTracker):
    def run(self, boxes, embeddings, **kwargs):
        """
        Runs tracking for current list of boxes, this is based only on proximity of the previous bboxes
        Args:
            **kwargs:
            embeddings: List<ndarray> of embeddings for cropped bboxes
            boxes: Dict<bbox: class_name>

        Returns: None

        """
        boxes = pre_process_boxes(boxes)
        boxes = np.array(boxes)
        if boxes.shape[0] == 0:
            return None

        if len(self.tracks) < boxes.shape[0]:
            for i in range(len(self.tracks), boxes.shape[0]):
                self.tracks.append(SiameseTrack(i))

        distance_to_tracks = np.empty((boxes.shape[0], len(self.tracks)))
        for embedding_id, embedding in enumerate(embeddings):
            for track_id, track in enumerate(self.tracks):
                distance_to_tracks[embedding_id][track_id] = track.get_similarity_to_embedding(embedding)

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
