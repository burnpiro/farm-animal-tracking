import matplotlib.pyplot as plt
import os

import numpy as np

import sys

if len(sys.argv)!=2:
    print(f'USAGE: {sys.argv[0]} <path_to_image>')
    exit()

import tensorflow as tf
from PIL import Image

from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder

from helpers.bb_helper import get_bb

from imageio import imread

if not os.path.isdir(f'crop_images'):
    os.mkdir(f'crop_images')

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


pipeline_config = 'model/inference_graph/pipeline.config'
model_dir = 'model/detection_model/inference_graph/checkpoint'

configs = config_util.get_configs_from_pipeline_file(pipeline_config)
model_config = configs['model']
detection_model = model_builder.build(
    model_config=model_config, is_training=False)

ckpt = tf.compat.v2.train.Checkpoint(
    model=detection_model)
ckpt.restore(os.path.join(model_dir, 'ckpt-0')).expect_partial()


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


detect_fn = get_model_detection_function(detection_model)


image_np = imread(sys.argv[1]).astype('uint8')
input_tensor = tf.convert_to_tensor(
    np.expand_dims(image_np, 0), dtype=tf.float32)
detections, predictions_dict, shapes = detect_fn(input_tensor)

label_id_offset = 1
image_np_with_detections = image_np.copy()

keypoints, keypoint_scores = None, None
if 'detection_keypoints' in detections:
    keypoints = detections['detection_keypoints'][0].numpy()
    keypoint_scores = detections['detection_keypoint_scores'][0].numpy()


label_map_path = os.path.join("model/detection_model/", configs['eval_input_config'].label_map_path)
label_map = label_map_util.load_labelmap(label_map_path)
categories = label_map_util.convert_label_map_to_categories(
    label_map,
    max_num_classes=label_map_util.get_max_label_map_index(label_map),
    use_display_name=True)
category_index = label_map_util.create_category_index(categories)
label_map_dict = label_map_util.get_label_map_dict(
    label_map, use_display_name=True)

viz_utils.visualize_boxes_and_labels_on_image_array(
    image_np_with_detections,
    detections['detection_boxes'][0].numpy(),
    (detections['detection_classes'][0].numpy() + label_id_offset).astype(int),
    detections['detection_scores'][0].numpy(),
    category_index,
    use_normalized_coordinates=True,
    max_boxes_to_draw=200,
    min_score_thresh=.50,
    agnostic_mode=False,
    keypoints=keypoints,
    keypoint_scores=keypoint_scores,
    keypoint_edges=get_keypoint_tuples(configs['eval_config']))

boxes = get_bb(
    image_np_with_detections,
    detections['detection_boxes'][0].numpy(),
    (detections['detection_classes'][0].numpy() + label_id_offset).astype(int),
    detections['detection_scores'][0].numpy(),
    category_index,
    use_normalized_coordinates=True,
    max_boxes_to_draw=200,
    min_score_thresh=.50,
    agnostic_mode=False,
    keypoints=keypoints,
    keypoint_scores=keypoint_scores,
    keypoint_edges=get_keypoint_tuples(configs['eval_config']))

for idx, (box, _) in enumerate(boxes.items()):
    ymin, xmin, ymax, xmax = box
    bb_image = tf.image.crop_to_bounding_box(
        input_tensor[0],
        int(ymin*input_tensor[0].shape[0]),
        int(xmin*input_tensor[0].shape[1]),
        int((ymax - ymin)*input_tensor[0].shape[0]),
        int((xmax - xmin)*input_tensor[0].shape[1])
    )
    image = Image.fromarray(tf.cast(bb_image, tf.uint8).numpy())
    image.save(f"crop_images/{idx}.jpg")


plt.figure(figsize=(12, 16))
plt.imshow(image_np_with_detections)
plt.savefig('crop_images/out.png')
