# Reference https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb

import numpy as np
import tensorflow as tf

MODEL_PATH = 'model/ssd_mobilenet_v1_coco_2018_01_28/'
PATH_TO_FROZEN_GRAPH = MODEL_PATH + 'frozen_inference_graph.pb'

# Load frozen TF model
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

# TODO: Incomplete code. Just testing some stuff