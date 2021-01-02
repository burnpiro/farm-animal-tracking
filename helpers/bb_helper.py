import collections
import numpy as np
from statistics import mean

import six

STANDARD_COLORS = [
    "AliceBlue",
    "Chartreuse",
    "Aqua",
    "Aquamarine",
    "Azure",
    "Beige",
    "Bisque",
    "BlanchedAlmond",
    "BlueViolet",
    "BurlyWood",
    "CadetBlue",
    "AntiqueWhite",
    "Chocolate",
    "Coral",
    "CornflowerBlue",
    "Cornsilk",
    "Crimson",
    "Cyan",
    "DarkCyan",
    "DarkGoldenRod",
    "DarkGrey",
    "DarkKhaki",
    "DarkOrange",
    "DarkOrchid",
    "DarkSalmon",
    "DarkSeaGreen",
    "DarkTurquoise",
    "DarkViolet",
    "DeepPink",
    "DeepSkyBlue",
    "DodgerBlue",
    "FireBrick",
    "FloralWhite",
    "ForestGreen",
    "Fuchsia",
    "Gainsboro",
    "GhostWhite",
    "Gold",
    "GoldenRod",
    "Salmon",
    "Tan",
    "HoneyDew",
    "HotPink",
    "IndianRed",
    "Ivory",
    "Khaki",
    "Lavender",
    "LavenderBlush",
    "LawnGreen",
    "LemonChiffon",
    "LightBlue",
    "LightCoral",
    "LightCyan",
    "LightGoldenRodYellow",
    "LightGray",
    "LightGrey",
    "LightGreen",
    "LightPink",
    "LightSalmon",
    "LightSeaGreen",
    "LightSkyBlue",
    "LightSlateGray",
    "LightSlateGrey",
    "LightSteelBlue",
    "LightYellow",
    "Lime",
    "LimeGreen",
    "Linen",
    "Magenta",
    "MediumAquaMarine",
    "MediumOrchid",
    "MediumPurple",
    "MediumSeaGreen",
    "MediumSlateBlue",
    "MediumSpringGreen",
    "MediumTurquoise",
    "MediumVioletRed",
    "MintCream",
    "MistyRose",
    "Moccasin",
    "NavajoWhite",
    "OldLace",
    "Olive",
    "OliveDrab",
    "Orange",
    "OrangeRed",
    "Orchid",
    "PaleGoldenRod",
    "PaleGreen",
    "PaleTurquoise",
    "PaleVioletRed",
    "PapayaWhip",
    "PeachPuff",
    "Peru",
    "Pink",
    "Plum",
    "PowderBlue",
    "Purple",
    "Red",
    "RosyBrown",
    "RoyalBlue",
    "SaddleBrown",
    "Green",
    "SandyBrown",
    "SeaGreen",
    "SeaShell",
    "Sienna",
    "Silver",
    "SkyBlue",
    "SlateBlue",
    "SlateGray",
    "SlateGrey",
    "Snow",
    "SpringGreen",
    "SteelBlue",
    "GreenYellow",
    "Teal",
    "Thistle",
    "Tomato",
    "Turquoise",
    "Violet",
    "Wheat",
    "White",
    "WhiteSmoke",
    "Yellow",
    "YellowGreen",
]


def _get_multiplier_for_color_randomness():
    """Returns a multiplier to get semi-random colors from successive indices.

    This function computes a prime number, p, in the range [2, 17] that:
    - is closest to len(STANDARD_COLORS) / 10
    - does not divide len(STANDARD_COLORS)

    If no prime numbers in that range satisfy the constraints, p is returned as 1.

    Once p is established, it can be used as a multiplier to select
    non-consecutive colors from STANDARD_COLORS:
    colors = [(p * i) % len(STANDARD_COLORS) for i in range(20)]
    """
    num_colors = len(STANDARD_COLORS)
    prime_candidates = [5, 7, 11, 13, 17]

    # Remove all prime candidates that divide the number of colors.
    prime_candidates = [p for p in prime_candidates if num_colors % p]
    if not prime_candidates:
        return 1

    # Return the closest prime number to num_colors / 10.
    abs_distance = [np.abs(num_colors / 10.0 - p) for p in prime_candidates]
    num_candidates = len(abs_distance)
    inds = [i for _, i in sorted(zip(abs_distance, range(num_candidates)))]
    return prime_candidates[inds[0]]


def get_bb(
    image,
    boxes,
    classes,
    scores,
    category_index,
    instance_masks=None,
    instance_boundaries=None,
    keypoints=None,
    keypoint_scores=None,
    keypoint_edges=None,
    track_ids=None,
    use_normalized_coordinates=False,
    max_boxes_to_draw=20,
    min_score_thresh=0.5,
    agnostic_mode=False,
    line_thickness=4,
    groundtruth_box_visualization_color="black",
    skip_boxes=False,
    skip_scores=False,
    skip_labels=False,
    skip_track_ids=False,
):
    """Overlay labeled boxes on an image with formatted scores and label names.

    This function groups boxes that correspond to the same location
    and creates a display string for each detection and overlays these
    on the image. Note that this function modifies the image in place, and returns
    that same image.

    Args:
      image: uint8 numpy array with shape (img_height, img_width, 3)
      boxes: a numpy array of shape [N, 4]
      classes: a numpy array of shape [N]. Note that class indices are 1-based,
        and match the keys in the label map.
      scores: a numpy array of shape [N] or None.  If scores=None, then
        this function assumes that the boxes to be plotted are groundtruth
        boxes and plot all boxes as black with no classes or scores.
      category_index: a dict containing category dictionaries (each holding
        category index `id` and category name `name`) keyed by category indices.
      instance_masks: a uint8 numpy array of shape [N, image_height, image_width],
        can be None.
      instance_boundaries: a numpy array of shape [N, image_height, image_width]
        with values ranging between 0 and 1, can be None.
      keypoints: a numpy array of shape [N, num_keypoints, 2], can
        be None.
      keypoint_scores: a numpy array of shape [N, num_keypoints], can be None.
      keypoint_edges: A list of tuples with keypoint indices that specify which
        keypoints should be connected by an edge, e.g. [(0, 1), (2, 4)] draws
        edges from keypoint 0 to 1 and from keypoint 2 to 4.
      track_ids: a numpy array of shape [N] with unique track ids. If provided,
        color-coding of boxes will be determined by these ids, and not the class
        indices.
      use_normalized_coordinates: whether boxes is to be interpreted as
        normalized coordinates or not.
      max_boxes_to_draw: maximum number of boxes to visualize.  If None, draw
        all boxes.
      min_score_thresh: minimum score threshold for a box or keypoint to be
        visualized.
      agnostic_mode: boolean (default: False) controlling whether to evaluate in
        class-agnostic mode or not.  This mode will display scores but ignore
        classes.
      line_thickness: integer (default: 4) controlling line width of the boxes.
      groundtruth_box_visualization_color: box color for visualizing groundtruth
        boxes
      skip_boxes: whether to skip the drawing of bounding boxes.
      skip_scores: whether to skip score when drawing a single detection
      skip_labels: whether to skip label when drawing a single detection
      skip_track_ids: whether to skip track id when drawing a single detection

    Returns:
      uint8 numpy array with shape (img_height, img_width, 3) with overlaid boxes.
    """
    # Create a display string (and color) for every box location, group any boxes
    # that correspond to the same location.
    box_to_display_str_map = collections.defaultdict(list)
    box_to_color_map = collections.defaultdict(str)
    box_to_instance_masks_map = {}
    box_to_instance_boundaries_map = {}
    box_to_keypoints_map = collections.defaultdict(list)
    box_to_keypoint_scores_map = collections.defaultdict(list)
    box_to_track_ids_map = {}
    if not max_boxes_to_draw:
        max_boxes_to_draw = boxes.shape[0]
    for i in range(boxes.shape[0]):
        if max_boxes_to_draw == len(box_to_color_map):
            break
        if scores is None or scores[i] > min_score_thresh:
            box = tuple(boxes[i].tolist())
            if instance_masks is not None:
                box_to_instance_masks_map[box] = instance_masks[i]
            if instance_boundaries is not None:
                box_to_instance_boundaries_map[box] = instance_boundaries[i]
            if keypoints is not None:
                box_to_keypoints_map[box].extend(keypoints[i])
            if keypoint_scores is not None:
                box_to_keypoint_scores_map[box].extend(keypoint_scores[i])
            if track_ids is not None:
                box_to_track_ids_map[box] = track_ids[i]
            if scores is None:
                box_to_color_map[box] = groundtruth_box_visualization_color
            else:
                display_str = ""
                if not skip_labels:
                    if not agnostic_mode:
                        if classes[i] in six.viewkeys(category_index):
                            class_name = category_index[classes[i]]["name"]
                        else:
                            class_name = "N/A"
                        display_str = str(class_name)
                if not skip_scores:
                    if not display_str:
                        display_str = "{}%".format(round(100 * scores[i]))
                    else:
                        display_str = "{}: {}%".format(
                            display_str, round(100 * scores[i])
                        )
                if not skip_track_ids and track_ids is not None:
                    if not display_str:
                        display_str = "ID {}".format(track_ids[i])
                    else:
                        display_str = "{}: ID {}".format(display_str, track_ids[i])
                box_to_display_str_map[box].append(display_str)
                if agnostic_mode:
                    box_to_color_map[box] = "DarkOrange"
                elif track_ids is not None:
                    prime_multipler = _get_multiplier_for_color_randomness()
                    box_to_color_map[box] = STANDARD_COLORS[
                        (prime_multipler * track_ids[i]) % len(STANDARD_COLORS)
                    ]
                else:
                    box_to_color_map[box] = STANDARD_COLORS[
                        classes[i] % len(STANDARD_COLORS)
                    ]

    return box_to_color_map


def pre_process_boxes(boxes):
    """
    Converts default boxes Dict into List of boxes
    Args:
        boxes: Dict<bbox: class_name>

    Returns: List<(y1,x1,y2,x2)>
        List of bboxes
    """
    new_boxes = []
    for idx, (box, _) in enumerate(boxes.items()):
        new_boxes.append(box)

    return new_boxes


def bbox_to_position(bbox):
    """
    Returns x,y position from bbox
    Args:
        bbox: List<(y1,x1,y2,x2)>

    Returns: List<(x,y)>

    """
    return [float(mean([bbox[1], bbox[3]])), float(mean([bbox[0], bbox[2]]))]
