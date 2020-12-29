import numpy as np
import codecs
import cv2
import json
import os
from multiprocessing import Pool
from statistics import mean

from scipy.io import loadmat

DATASET_DIR = "../PigTrackingDataset2020"
videos_dir = os.path.join(DATASET_DIR, "videos")
mat_file = os.path.join(DATASET_DIR, "LocationSelected.mat")
labels_mat_file = os.path.join(videos_dir, "InitialLabels.mat")
folder_prefix = "tracking"
images_folder_prefix = "images"


def clamp(x):
    """
    Clamps data to be within <0,1> values
    Args:
        x: float

    Returns: float

    """
    if x < 0:
        return 0
    if x > 1:
        return 1
    return x


def annotate(filename):
    """
    Creates two files for video in "filename":
    - frames_tracking.json - tracking data per video frame
    - pigs_tracking.json - tracking data per animal label on video

    frames_tracking.json structure:
    {
        "frame_id": List<x:float,y:float,id:int>
    }
    @x and @y are floats describing proportional position of the center point of the animal (ranges from 0 to 1).
    @id is an integer identifier for an animal


    pigs_tracking.json structure:
    {
        "animal_id": List<x:float, y:float, frame_id:int>
    }
    @x and @y are floats describing proportional position of the center point of the animal (ranges from 0 to 1).
    @frame_id is a frame id (incremental)

    Args:
        filename: string path to video

    Returns:

    """
    cap = cv2.VideoCapture(filename)
    index = int(os.path.basename(filename)[:2]) - 1
    mat = loadmat(mat_file)["LocationSelected"][0][index]
    labels = loadmat(labels_mat_file)["InitialLabels"][0][index][0]

    # Create directory for current video id if there is none
    if not os.path.isdir(f"{folder_prefix}/{index+1:02d}"):
        os.mkdir(f"{folder_prefix}/{index+1:02d}")

    i = 0
    frames_annotations = {}
    pigs_positions = {}
    frames_missing = 0

    while cap.isOpened():
        ret, frame = cap.read()

        """
        Check if frame is missing, if there is more than 5 frames missing break the loop (corrupted video?)
        """
        if frame is None:
            frames_missing += 1
            if frames_missing > 5:
                raise Exception('Video might be corrupted, more than 5 frames is missing')
            else:
                continue

        # Also break when current frame id exceed annotations in matlab file
        if i >= mat.shape[1]:
            raise Exception('.mat file might be invalid, there is more frames in the video than defined in annotations')

        # List of position for frames
        aa_positions = []

        # Loop through every annotated animal in the video
        for j in range(mat.shape[-1]):
            # Get definition of two points for current frame and animal [x1, y1, x2, y2]
            points = mat[:, i, j]

            # If there is no list for current animal, create one
            if f"{labels[j]}" not in pigs_positions:
                pigs_positions[f"{labels[j]}"] = []

            # Append scaled mean of two points (center of the animal)
            aa_positions.append(
                (
                    float(mean([points[0], points[2]]) / frame.shape[1]),
                    float(mean([points[1], points[3]]) / frame.shape[0]),
                    int(labels[j]),
                )
            )
            pigs_positions[f"{labels[j]}"].append(
                (
                    float(mean([points[0], points[2]]) / frame.shape[1]),
                    float(mean([points[1], points[3]]) / frame.shape[0]),
                    int(i),
                )
            )

        # assign frame annotations to current frame
        frames_annotations[f"{i}"] = aa_positions
        i += 1

    # dump all positions to corresponding json files
    json.dump(
        frames_annotations,
        codecs.open(
            f"{folder_prefix}/{index+1:02d}/frames_tracking.json", "w", encoding="utf-8"
        ),
        separators=(",", ":"),
    )
    json.dump(
        pigs_positions,
        codecs.open(
            f"{folder_prefix}/{index+1:02d}/pigs_tracking.json", "w", encoding="utf-8"
        ),
        separators=(",", ":"),
    )

    cap.release()


if __name__ == "__main__":
    """
        Create multiple processes to generate tracking data
    """
    numberOfThreads = 4
    if not os.path.isdir(f"{folder_prefix}"):
        os.mkdir(f"{folder_prefix}")

    pool = Pool(processes=numberOfThreads)
    entires = []
    for entry in os.listdir(videos_dir):
        if (
            entry.endswith(".mp4")
            and "_annotated" not in entry
            and "_tracking_viz" not in entry
            and "_seg" not in entry
        ):
            print(f"processing {os.path.join(videos_dir, entry)}")
            entires.append(os.path.join(videos_dir, entry))
    pool.map(annotate, entires)

    cv2.destroyAllWindows()
