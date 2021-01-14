from abc import ABC
import numpy as np
import os
import cv2
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
from helpers.bb_helper import get_bb


dirname = os.path.dirname(__file__)
pipeline_config = os.path.join(dirname, "inference_graph/pipeline.config")
model_dir = os.path.join(dirname, "inference_graph/checkpoint")


def get_keypoint_tuples(eval_config):
    """Return a tuple list of keypoint edges from the eval config.

    Args:
      eval_config: an eval config containing the keypoint edges

    Returns:
      a list of edge tuples, each in the format (start, end)
    """
    tuple_list = []
    kp_list = eval_config.keypoint_edge
    for edge in kp_list:
        tuple_list.append((edge.start, edge.end))
    return tuple_list


def get_model_detection_function(model):
    """Get a tf.function for detection."""

    @tf.function
    def detect_fn(image):
        """Detect objects in image."""

        image, shapes = model.preprocess(image)
        prediction_dict = model.predict(image, shapes)
        detections = model.postprocess(prediction_dict, shapes)

        return detections, prediction_dict, tf.reshape(shapes, [-1])

    return detect_fn


class DefaultDetectionModel(ABC):
    """
    Default detection model. It wraps TF model and provides easier to understand API.
    """

    def __init__(self, config_path=pipeline_config, checkpoint_dir=model_dir):
        super().__init__()
        self.label_id_offset = 1
        self.configs = config_util.get_configs_from_pipeline_file(config_path)
        self.model_config = self.configs["model"]
        self.detection_model = model_builder.build(
            model_config=self.model_config, is_training=False
        )

        ckpt = tf.compat.v2.train.Checkpoint(model=self.detection_model)
        ckpt.restore(os.path.join(checkpoint_dir, "ckpt-0")).expect_partial()
        self.detect_fn = get_model_detection_function(self.detection_model)

        label_map_path = os.path.join(
            dirname, self.configs["eval_input_config"].label_map_path
        )
        label_map = label_map_util.load_labelmap(label_map_path)
        categories = label_map_util.convert_label_map_to_categories(
            label_map,
            max_num_classes=label_map_util.get_max_label_map_index(label_map),
            use_display_name=True,
        )
        self.category_index = label_map_util.create_category_index(categories)

    def predict(self, image_np: np.ndarray):
        """
        Runs detection on input image and return dict with bounding boxes
        Args:
            image_np: np.ndarray, Image as np array

        Returns: Dict<box, class>
            Dictionary with BB as keys
        """
        single_image = image_np.ndim == 3
        if single_image:
            image_np = np.expand_dims(image_np, 0)
        input_tensor = tf.convert_to_tensor(
            image_np, dtype=tf.float32
        )

        detections, predictions_dict, shapes = self.detect_fn(input_tensor)

        boxes = []

        for i in range(image_np.shape[0]):
            boxes.append(get_bb(
                image_np[i],
                detections["detection_boxes"][i].numpy(),
                (detections["detection_classes"][i].numpy() + self.label_id_offset).astype(
                    int
                ),
                detections["detection_scores"][i].numpy(),
                self.category_index,
                use_normalized_coordinates=True,
                max_boxes_to_draw=200,
                min_score_thresh=0.50,
                agnostic_mode=False,
                keypoints=None,
                keypoint_scores=None,
                keypoint_edges=get_keypoint_tuples(
                    self.configs["eval_config"]),
            ))
        if single_image:
            return boxes[0]
        return boxes

    @staticmethod
    def draw_bb(image_np: np.ndarray, boxes: dict):
        """
        Draws boxes on the image
        Args:
            image_np: np.ndarray, Image as np array
            boxes: Dict<box, class> - Dictionary with BB as keys

        Returns: np.ndarray Image with boxes
        """
        result = image_np.copy()
        width = result.shape[1]
        height = result.shape[0]

        for idx, (box, _) in enumerate(boxes.items()):
            ymin, xmin, ymax, xmax = box
            result = cv2.rectangle(
                result,
                (int(xmin * width), int(ymin * height)),
                (int(xmax * width), int(ymax * height)),
                (0, 255, 0),
                2,
            )

        return result

    @staticmethod
    def crop_bb(image_np: np.ndarray, boxes: dict):
        """
        Crop BB from image base on "boxes" definition
        Args:
            image_np: np.ndarray, Image as np array
            boxes: Dict<box, class> - Dictionary with BB as keys

        Returns: List<Tensor>
            List of cropped images in TF tensor format
        """
        input_tensor = tf.convert_to_tensor(
            np.expand_dims(image_np, 0), dtype=tf.float32
        )
        images = []
        for idx, (box, _) in enumerate(boxes.items()):
            ymin, xmin, ymax, xmax = box
            bb_image = tf.image.crop_to_bounding_box(
                input_tensor[0],
                int(ymin * input_tensor[0].shape[0]),
                int(xmin * input_tensor[0].shape[1]),
                int((ymax - ymin) * input_tensor[0].shape[0]),
                int((xmax - xmin) * input_tensor[0].shape[1]),
            )
            images.append(bb_image)

        return images
