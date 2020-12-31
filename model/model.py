import numpy as np
from imageio import imread
import cv2
from tqdm import tqdm
from model.abstract_model import AbstractModel
from model.detection_model.detection_model import DefaultDetectionModel
from model.siamese.siamese_model import DefaultSiameseModel
from model.tracker.tracker import Tracker


class Model(AbstractModel):
    def __init__(
        self,
        detection_model: DefaultDetectionModel,
        recognition_model: DefaultSiameseModel,
        tracker: Tracker,
    ):
        """
        Args:
            detection_model: DefaultDetectionModel
            recognition_model: DefaultSiameseModel
            tracker: Tracker
        """
        super().__init__()
        self.detection_model = detection_model
        self.recognition_model = recognition_model
        self.tracker = tracker

    def predict_image(self, path_to_img: str):
        """

        Args:
            path_to_img: str - path to image

        Returns: Dict<object_id, Tuple(x,y)>
            Predictions for all objects on the image.
        """
        image_np = imread(path_to_img).astype("uint8")
        boxes = self.detection_model.predict(image_np)
        cropped_images = DefaultDetectionModel.crop_bb(image_np, boxes)
        embeddings = self.recognition_model.predict(cropped_images)

        return embeddings

    def predict_video(self, path_to_video: str, return_type="objets"):
        """

        Args:
            path_to_video:
            return_type: string - type of return objects, either "frames" or "objects"
                "frames" - returns predictions in form of the list where each element is a prediction for one frame
                "objects" - returns predictions in form of the dict, each element is a list of predictions for given
                            object_id

        Returns: List<Dict<object_id, Tuple(x,y)>> or Dict<object_id, List<x,y>>
            Depends on "return_type" value
        """
        cap = cv2.VideoCapture(path_to_video)

        i = 0
        num_of_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        pbar = tqdm(total=num_of_frames)

        while cap.isOpened():
            ret, frame = cap.read()
            i += 1
            pbar.update(i)
            if frame is None:
                break

            frame = frame.astype("uint8")
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            boxes = self.detection_model.predict(rgb_frame)
            cropped_images = DefaultDetectionModel.crop_bb(rgb_frame, boxes)
            embeddings = self.recognition_model.predict(cropped_images)
            self.tracker.run(boxes, embeddings)

        pbar.close()
        cap.release()
        cv2.destroyAllWindows()

        return {
            track.track_id: np.array(track.history) for track in self.tracker.tracks
        }

    def print_bb_on_image(self, path_to_img: str):
        """
        Runs object detection and prints bbs on the image
        Args:
            path_to_img: str - path to image

        Returns: np.ndarray
            Image with boxes
        """
        image_np = imread(path_to_img).astype("uint8")
        boxes = self.detection_model.predict(image_np)
        result = DefaultDetectionModel.draw_bb(image_np, boxes)
        return result

    def recognize_animals_on_image(self, path_to_img: str):
        """
        Runs object recognition and prints them on the image
        Args:
            path_to_img: str - path to image

        Returns: np.ndarray
            Image with object ids
        """
        pass
