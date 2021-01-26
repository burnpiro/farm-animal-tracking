import cv2
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

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
                "avg_err": track["total"] / len(paths[track_id]),
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


def cm_analysis(cm, labels, figsize=(30,30), filename="conf_matrix.png"):
    """
    Generate matrix plot of confusion matrix with pretty annotations.
    The plot image is saved to disk.
    args:
      cm:    confusion matrix
      filename:  filename of figure file to save (has to be PNG)
      labels:    string array, name the order of class labels in the confusion matrix.
                 use `clf.classes_` if using scikit-learn models.
                 with shape (nclass,).
      figsize:   the size of the figure plotted.
    """
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if i == j:
                s = cm_sum[i]
                annot[i, j] = '%.1f%%\n%d' % (p, c)
            elif c == 0:
                annot[i, j] = '0'
            else:
                annot[i, j] = '%.1f%%\n%d' % (p, c)
    cm = pd.DataFrame(cm, index=labels, columns=labels)
    cm.index.name = 'Actual'
    cm.columns.name = 'Predicted'
    fig, ax = plt.subplots(figsize=figsize)
    fig.suptitle('Confusion Matrix', fontsize=42)
    ax.xaxis.label.set_size(36)
    ax.yaxis.label.set_size(36)
    ax.tick_params(axis='both', which='major', labelsize=26)
    sns.set(font_scale=2.0)
    sns.heatmap(cm, cmap="viridis", annot=annot, fmt='', ax=ax)
    plt.savefig(filename, transparent=True)
    # plt.show()


def classification_report_latex(report, filename="report.txt"):
    df = pd.DataFrame(report).transpose()
    with open(filename, 'w') as tf:
        tf.write(df.to_latex())
