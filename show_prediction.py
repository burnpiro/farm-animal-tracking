import numpy as np
import cv2
import json
import os
import sys

if len(sys.argv)!=2:
    print(f'USAGE: {sys.argv[0]} <path_to_video>')
    exit()

import tensorflow as tf
gpu = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpu[0], True)
tf.config.set_visible_devices([], 'GPU')

from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder

import time

treshold = 0.6


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
    pipeline_config = 'model/detection_model/inference_graph/pipeline.config'
    model_dir = 'model/detection_model/inference_graph/checkpoint'

    configs = config_util.get_configs_from_pipeline_file(pipeline_config)
    model_config = configs['model']
    detection_model = model_builder.build(
        model_config=model_config, is_training=False)

    ckpt = tf.compat.v2.train.Checkpoint(
        model=detection_model)
    ckpt.restore(os.path.join(model_dir, 'ckpt-0')).expect_partial()

    detect_fn = get_model_detection_function(detection_model)

    label_map_path = os.path.join("model/detection_model/", configs['eval_input_config'].label_map_path)
    label_map = label_map_util.load_labelmap(label_map_path)
    categories = label_map_util.convert_label_map_to_categories(
        label_map,
        max_num_classes=label_map_util.get_max_label_map_index(label_map),
        use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
    label_map_dict = label_map_util.get_label_map_dict(
        label_map, use_display_name=True)

    cap = cv2.VideoCapture(sys.argv[1])

    i = 0
    frame_start = 0
    import matplotlib.pyplot as plt
    while(cap.isOpened()):
        frame_end = time.time()
        print(f'fps: {1/(frame_end-frame_start)}')
        frame_start = time.time()
        ret, frame = cap.read()
        i += 1

        frame = frame.astype('uint8')

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_tensor = tf.convert_to_tensor(
            np.expand_dims(rgb_frame, 0), dtype=tf.float32)


        detections, predictions_dict, shapes = detect_fn(input_tensor)

        boxes = detections['detection_boxes'][0]
        scores = detections['detection_scores'][0]

        resized = cv2.resize(frame, (800, 600))

        viz_utils.visualize_boxes_and_labels_on_image_array(
            resized,
            detections['detection_boxes'][0].numpy(),
            (detections['detection_classes'][0].numpy() + 1).astype(int),
            detections['detection_scores'][0].numpy(),
            category_index,
            use_normalized_coordinates=True,
            max_boxes_to_draw=200,
            min_score_thresh=.30,
            agnostic_mode=False,
            keypoints=None,
            keypoint_scores=None,
            keypoint_edges=get_keypoint_tuples(configs['eval_config']))

        # j = 0
        # while scores[j] > treshold:
        #     j += 1
        #     box = boxes[j]
        #     cv2.rectangle(resized, (int(box[1]*800), int(box[0]*600)),
        #                   (int(box[3]*800), int(box[2]*600)), (255, 0, 255), 2)

        cv2.imshow('frame', resized)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
