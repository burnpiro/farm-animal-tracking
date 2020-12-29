import numpy as np
import cv2
import json
import os

from PIL import Image, ImageDraw
from scipy.io import loadmat


if __name__ == '__main__':
    cap = cv2.VideoCapture('../PigTrackingDataset2020/videos/01_early_finisher_high_activity_day.mp4')
    mat = loadmat('../PigTrackingDataset2020/LocationSelected.mat')['LocationSelected'][0][0]

    i = 0
    while(cap.isOpened()):
        ret, frame = cap.read()

        if i >= mat.shape[1]:
            break

        for j in range(mat.shape[-1]):
            point = mat[:, i, j]

            cv2.circle(frame, center=tuple(point[:2].astype(
                np.int32)), radius=20, color=(255, 0, 0), thickness=-1)
            cv2.circle(frame, center=tuple(point[2:].astype(
                np.int32)), radius=20, color=(0, 255, 0), thickness=-1)

            pos3 = ((point[:2] - point[2:])*1.4+point[2:]).astype(np.int32)
            pos4 = ((point[:2] - point[2:])*-0.8+point[2:]).astype(np.int32)

            v = (point[:2] - point[2:])[::-1] * 0.4

            x1 = min([(pos3+v)[0], (pos3-v)[0], (pos4+v)[0], (pos4-v)[0]])
            x2 = max([(pos3+v)[0], (pos3-v)[0], (pos4+v)[0], (pos4-v)[0]])

            y1 = min([(pos3+v)[1], (pos3-v)[1], (pos4+v)[1], (pos4-v)[1]])
            y2 = max([(pos3+v)[1], (pos3-v)[1], (pos4+v)[1], (pos4-v)[1]])

            cv2.circle(frame, center=tuple(pos3), radius=20, color=(0, 0, 255), thickness=-1)
            cv2.circle(frame, center=tuple(pos4), radius=20, color=(255, 0, 255), thickness=-1)

            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 255), 2)

        resized = cv2.resize(frame, (800, 600))
        cv2.imshow('frame', resized)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        i += 1

    cap.release()
    cv2.destroyAllWindows()
