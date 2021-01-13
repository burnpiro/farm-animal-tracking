import cv2
import os
from data.evaluator import Evaluator


def extract_scores(scores, paths):
    """
    Args:
        scores: {
            [string]: {
                total: [float]
                intervals?: {
                    interval: [int],
                    part: List[float]
                }
            }
        }
        paths: Dict<string, List[x,y]>

    Returns:
        {
            [string]: {
                total: {
                    "abs_err": float,
                    "mea": float
                },
                intervals?: {
                    interval: [int],
                    parts: List[float]
                }
            }
        }
    """
    new_scores = {}

    for track_id, track in scores.items():
        new_scores[track_id] = {
            "total": {
                "abs_err": track["total"],
                "mae": track["total"] / len(paths[track_id]),
            }
        }

        if "intervals" in track:
            new_scores[track_id]["intervals"] = track["intervals"]

    return new_scores


def print_path_comparison(
    out_dir: str, annotation, path, obj_id: int, interval=None, parts=None
):
    """
    Prints compared paths on image
    Args:
        out_dir: [string] - directory to output to
        annotation: List[x,y]
        path: List[x,y]
        obj_id: int
        interval: [int] - if scores provided set interval for comparison
        parts: List[float] - list of scores for interval comparison

    Returns:
        None
    """
    cv2.imwrite(
        os.path.join(out_dir, f"{obj_id}_compare.jpg"),
        Evaluator.draw_paths_comparison(annotation, path),
    )
    if parts is not None:
        cv2.imwrite(
            os.path.join(out_dir, f"{obj_id}_partial_compare.jpg"),
            Evaluator.draw_path_parts_comparison(annotation, path, parts, interval),
        )
