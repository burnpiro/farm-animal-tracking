import numpy as np
import cv2
import operator
import uuid

from scipy.spatial.distance import euclidean, cosine
from helpers.bb_helper import pre_process_boxes, bbox_to_position
from model.tracker.embedding_track import EmbeddingTrack
from model.tracker.default_tracker import DefaultTracker

font = cv2.FONT_HERSHEY_SIMPLEX


class EmbeddingTracker(DefaultTracker):
    def __init__(self, labels: list = None, vectors_path="", meta_path="", interval=10, max_jump=0.2) -> None:
        """
        Args:
            labels: List of labels for tracking
            vectors_path: path to vecs-conc-MODEL_NAME.tsv file
            meta_path: path to meta-conc-MODEL_NAME.tsv file
            interval: how many frames to use when calculating distance from cluster center
            max_jump: max distance between frames (in relation to image diagonal)
        """
        super().__init__(labels)
        if vectors_path == "" or meta_path == "":
            raise Exception("Please provide paths with vectors and meta values for tracker")

        self.tempTracks = []
        self.currentInterval = 0
        self.interval = interval
        self.max_jump = max_jump
        self.avg_vectors = {}
        self.set_avg_vectors(vectors_path, meta_path)

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

    def clear_temp_tracks(self):
        self.tempTracks = []

    def get_track_class(self, track: EmbeddingTrack):
        return track.predicted_class

    def increase_curr_interval(self):
        self.currentInterval += 1

    def assign_temp_tracks_to_tracks(self):
        track_classes = {}
        for track in self.tempTracks:
            track_classes[track.track_id] = {}
            for embedding in track.embeddings:
                best_class, distances = self.compare_mean_with_vectors(embedding)
                print( best_class, distances, track.track_id)
                if best_class not in track_classes[track.track_id]:
                    track_classes[track.track_id][best_class] = 0
                track_classes[track.track_id][best_class] += 1
            max_class = max(track_classes[track.track_id].items(), key=operator.itemgetter(1))[0]
            track.predicted_class = max_class

        print([track.predicted_class for track in self.tempTracks])
        assigned_tracks = []
        assigned_classes = []
        for current_track in self.tracks:
            for track in self.tempTracks:
                max_class = max(track_classes[track.track_id].items(), key=operator.itemgetter(1))[0]
                print(max_class)
                if current_track.predicted_class == max_class and track.track_id not in assigned_tracks:
                    # self.tempTracks[track_id].predicted_class = max_class
                    assigned_classes.append(max_class)
                    current_track.merge_with_track(track)
                    assigned_tracks.append(track.track_id)
                    break

        copied_classes = []
        if len(assigned_tracks) < len(self.tempTracks):
            for track in self.tempTracks:
                if track.track_id not in assigned_tracks:
                    print(track.track_id)
                    print(track_classes[track.track_id])
                    max_class = max(track_classes[track.track_id].items(), key=operator.itemgetter(1))[0]
                    track.predicted_class = max_class
                    copied_classes.append(max_class)
                    self.tracks.append(track)
                    assigned_tracks.append(track.track_id)

        print("CALLED")
        print(assigned_tracks)
        print(assigned_classes)
        print(copied_classes)
        print([(track.history, track.predicted_class) for track in self.tracks])
        if assigned_classes:
            raise

        self.currentInterval = 0
        self.clear_temp_tracks()

    def compare_mean_with_vectors(self, mean):
        best_class = None
        best_distance = None
        distances = {}
        for class_id, vector in self.avg_vectors.items():
            distance = cosine(vector, mean)
            distances[class_id] = distance
            if best_distance is None or best_distance > distance:
                best_distance = distance
                best_class = class_id

        return best_class,distances

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
        self.increase_curr_interval()
        if self.currentInterval >= 10:
            self.assign_temp_tracks_to_tracks()

        if boxes.shape[0] == 0:
            return None

        if len(self.tempTracks) == 0:
            for i in range(len(self.tempTracks), boxes.shape[0]):
                self.tempTracks.append(EmbeddingTrack(uuid.uuid4()))

        distance_to_tracks = np.empty((boxes.shape[0], len(self.tempTracks)))
        for box_id, bbox in enumerate(boxes):
            for track_id, track in enumerate(self.tempTracks):
                distance_to_tracks[box_id][track_id] = track.get_distance_to_box(bbox)

        updated_tracks = []
        while np.min(distance_to_tracks) != np.inf and len(updated_tracks) < len(self.tempTracks):
            smallest_distance_pos = np.unravel_index(
                distance_to_tracks.argmin(), distance_to_tracks.shape
            )
            self.tempTracks[smallest_distance_pos[1]].update(
                boxes[smallest_distance_pos[0]], embeddings[smallest_distance_pos[0]]
            )
            self.tempTracks[smallest_distance_pos[1]].add_embedding(embeddings[smallest_distance_pos[0]])
            updated_tracks.append(smallest_distance_pos[1])
            distance_to_tracks[smallest_distance_pos[0], :] = np.inf
            distance_to_tracks[:, smallest_distance_pos[1]] = np.inf

        if len(updated_tracks) < len(self.tempTracks):
            for track_id, track in enumerate(self.tempTracks):
                if track_id not in updated_tracks:
                    track.update([-1, -1, -1, -1], None)
