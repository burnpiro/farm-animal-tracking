import numpy as np
import cv2
import json
import os
import multiprocessing

from PIL import Image, ImageDraw
from scipy.io import loadmat

DATASET_DIR = '../PigTrackingDataset2020'
mat_file = os.path.join(DATASET_DIR, 'LocationSelected.mat')
videos_dir = os.path.join(DATASET_DIR, 'videos')

def clamp(x):
    if x<0:
        return 0
    if x>1:
        return 1
    return x

def annotate(filename):    
    cap = cv2.VideoCapture(filename)
    index = int(os.path.basename(filename)[:2])-1
    mat = loadmat(mat_file)['LocationSelected'][0][index]

    os.mkdir(f'frames/{index+1:02d}')

    i = 0
    annotations = {}

    while(cap.isOpened()):
        ret, frame = cap.read()

        if i >= mat.shape[1]:
            break

        if i % 90 != 0:
            i += 1
            continue

        aabbs = []

        for j in range(mat.shape[-1]):
            point = mat[:, i, j]

            pos3 = ((point[:2] - point[2:])*1.4+point[2:]).astype(np.int32)
            pos4 = ((point[:2] - point[2:])*-0.8+point[2:]).astype(np.int32)

            v = (point[:2] - point[2:])[::-1] * 0.4

            x1 = min([(pos3+v)[0], (pos3-v)[0], (pos4+v)[0], (pos4-v)[0]])
            x2 = max([(pos3+v)[0], (pos3-v)[0], (pos4+v)[0], (pos4-v)[0]])

            y1 = min([(pos3+v)[1], (pos3-v)[1], (pos4+v)[1], (pos4-v)[1]])
            y2 = max([(pos3+v)[1], (pos3-v)[1], (pos4+v)[1], (pos4-v)[1]])

            aabbs.append((x1/frame.shape[1], y1/frame.shape[0], x2/frame.shape[1], y2/frame.shape[0]))

        name = f'frames/{index+1:02d}/frame{i:04d}.jpg'

        frame = cv2.resize(frame, (640, 640))
        cv2.imwrite(name, frame)
        annotations[name] = aabbs
        i += 1

    with open(f'frames/{index+1:02d}/annotations.json', 'w') as f:
        f.write(json.dumps(annotations))

    cap.release()


if __name__ == '__main__':
    processes = []
    os.mkdir('frames')
    for entry in os.listdir(videos_dir):
        if entry.endswith('.mp4'):
            p = multiprocessing.Process(target=annotate, args=(
                os.path.join(videos_dir, entry),))
            p.start()
            processes.append(p)
            
    for p in processes:
        p.join()
        
    cv2.destroyAllWindows()
