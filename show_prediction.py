import numpy as np
import cv2
import json
import os

import tensorflow as tf

from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder

treshold = 0.8


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


if __name__ == '__main__':
    pipeline_config = 'inference_graph/pipeline.config'
    model_dir = 'inference_graph/checkpoint'

    configs = config_util.get_configs_from_pipeline_file(pipeline_config)
    model_config = configs['model']
    detection_model = model_builder.build(
        model_config=model_config, is_training=False)

    ckpt = tf.compat.v2.train.Checkpoint(
        model=detection_model)
    ckpt.restore(os.path.join(model_dir, 'ckpt-0')).expect_partial()

    detect_fn = get_model_detection_function(detection_model)

    cap = cv2.VideoCapture(
        'PigTrackingDataset2020/videos/01_early_finisher_high_activity_day.mp4')

    i = 0
    while(cap.isOpened()):
        ret, frame = cap.read()
        i += 1

        frame = frame.astype('uint8')

        input_tensor = tf.convert_to_tensor(
            np.expand_dims(frame, 0), dtype=tf.float32)
        detections, predictions_dict, shapes = detect_fn(input_tensor)

        boxes = detections['detection_boxes'][0]
        scores = detections['detection_scores'][0]

        resized = cv2.resize(frame, (800, 600))

        j = 0
        while scores[j] > treshold:
            j += 1
            box = boxes[j]
            cv2.rectangle(resized, (int(box[1]*800), int(box[0]*600)),
                          (int(box[3]*800), int(box[2]*600)), (255, 0, 255), 2)

        cv2.imshow('frame', resized)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
