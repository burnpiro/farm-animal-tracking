# Farm Animal Tracking Project

Project for tracking farm animals.

## Dataset

Dataset for learning of model can be obtained at [PSRG website](http://psrg.unl.edu/Projects/Details/12-Animal-Tracking).

## EDA (Exploratory Data Analysis)

- Run:
```shell
docker-compose -f eda/docker-compose.yaml up
```
- Go to `localhost:8001` and enter token from console

## Model

1. To download precompiled model weights [Google Drive](https://drive.google.com/file/d/1OCkqrhT4LPlL4omqDQiw0_XzJ2J77F4n/view?usp=sharing)
1. Copy zip file into `./` directory
1. Run:

```
$ unzip inference_graph.tar.zip -d model
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

![](assets/prediction.png)

## Siamese network

You can download current best weights from [Firestore](https://firebasestorage.googleapis.com/v0/b/fat-farm-animal-tracking.appspot.com/o/siam-model-0.0012.h5?alt=media&token=ba006953-81f2-4c4a-bdfb-02607df12d7e). Put them into `./siamese/weights` and use the path as `--weights` parameter.

#### Training

Make sure you have cropped dataset in `./data/cropped_animals` folder. Please check `./data/data_generator.py` documentation for more info.

```
$ python train_siamese.py
```

#### Generate Embeddings for Test dataset an visualize it

```
$ python generate_siamese_emb_space.py
```

This is going to produce two files:

- vecs.tsv - list of embeddings for test dataset
- meta.tsv - list of labels for embeddings

You can visualize those embeddings in [https://projector.tensorflow.org/](https://projector.tensorflow.org/) application. Just upload them as a custom data (use `Load` option).

![](assets/emb-space.png)

#### Generate tracking data

```
$ cd data
$ python generate_tracking.py
```

This is going to produce tracking data from videos, so we can evaluate model. Look for `frames_tracking.json` and `pigs_tracking.json` inside `./data/tracking/`. For more details check Wiki.


### Testing two images

You can specify the weights for the model. Please use weights marked with the lowest number (loss value).

```
$ python test_siamese.py --source ./crop_images/5.jpg --target ./crop_images/1.jpg
```

Options:
```
--source ./crop_images/5.jpg
--target ./crop_images/1.jpg
--weights siam-model-0.0012.h5
```

