# Farm Animal Tracking Project

Project for tracking farm animals.

## Dataset

Dataset for learning of model can be obtained at [PSRG website](psrg.unl.edu/Projects/Details/12-Animal-Tracking).

## Model

To download precompiled model weights run:

```
$ wget https://srv-store5.gofile.io/download/kNljMT/inference_graph.tar.gz
$ tar zxvf inference_graph.tar.gz -C model
```

## Detection

To visualize animal detection on video use:
```
$ python show_prediction
```
or for image:
```
$ python run_detection.py
```

![](prediction.png)