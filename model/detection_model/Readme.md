# Model

To build model install object detection [library](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2.md).

Make sure dataset is located in home directory.
Download model files from [here](http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.tar.gz) and extract them in model directory.

<pre>
.
├── PigTrackingDataset2020
│   ├── LocationSelected.mat
│   └── videos
└──model
    ├── ssd_resnet50_v1_fpn_640x640_coco17_tpu-8
    ├── generate_frames.py
    └── generate_records.py
</pre>

Then in model directory run:

```
$ python generate_frames.py

$ python generate_records.py

$ python <location to object detection library>/model_main_tf2.py --pipeline_config_path="pipeline.config" --alsologtostderr --model_dir="resnet_model"
```

To export model graph to be used with application:

```
$ python <location to object detection library>/exporter_main_v2.py --input_type image_tensor --pipeline_config_path pipeline.config --trained_checkpoint_dir .\resnet_model\ --output_directory inference_graph
```