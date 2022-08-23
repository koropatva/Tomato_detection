import os
from pickle import FALSE
import sys
import random
import math
import re
import time
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
import skimage
import itertools
import logging
import json
import re
import random
from collections import OrderedDict
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as lines
from matplotlib.patches import Polygon

# Root directory of the project
ROOT_DIR = os.path.abspath("./")
print(os.listdir(ROOT_DIR))
# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log

MODEL_DIR = os.path.join(ROOT_DIR, "logs")

class TomatoConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "tomato"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + tomato
 
    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Skip detections with < 99% confidence
    DETECTION_MIN_CONFIDENCE = 0.99
    


class InferenceConfig(TomatoConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

def load_model():
    config = TomatoConfig()
    
    inference_config = InferenceConfig()

    # Recreate the model in inference mode
    model = modellib.MaskRCNN(mode="inference", 
                          config=inference_config,
                          model_dir=MODEL_DIR)

    # Get path to saved weights
    # Either set a specific path or find last trained weights
    model_path = os.path.join("/Users/a104133/Projects/Tomato_detection/logs/", "mask_rcnn_tomato.h5")
    #model_path = model.find_last()

    # Load trained weights
    print("Loading weights from ", model_path)
    model.load_weights(model_path, by_name=True)

    print("Loaded weights")
    return model

cap = cv2.VideoCapture('/Users/a104133/Downloads/TEST3.mov')

#[162, 107, 194, 158]
#x1, y1 = 130, 140
#x2, y2 = 170, 170

#width = x2 - x1
#height = y2 - y1

#search = 20

# Set up tracker
#tracker_types = ['MIL','KCF', 'CSRT']
#tracker_type = tracker_types[1]

#if tracker_type == 'MIL':
#    tracker = cv2.TrackerMIL_create()

#if tracker_type == 'KCF':
#    tracker = cv2.TrackerKCF_create()

#if tracker_type == "CSRT":
#    tracker = cv2.TrackerCSRT_create()

class Track:
    bbox = ()
    tracker = None
    ok = None
    color = ()

tracks = []
first = True

while cap.isOpened():
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    img = frame #cv2.resize(frame, (512, 512))

    if first:
        model = load_model()

        results = model.detect([img], verbose=1)
        
        r = results[0]
        
        print(r['rois'])

        for rois in r['rois']:
            y1, x1, y2, x2 = rois
            track = Track()
            track.bbox = (x1, y1, x2, y2)
            track.tracker = cv2.TrackerCSRT_create()
            track.ok = None
            track.color = (random.randint(10,250), random.randint(10,250), random.randint(10,250))
            tracks.append(track)
        first = False
        
    for track in tracks:
        if track.ok is None:
            track.ok = track.tracker.init(img, track.bbox)

        ok, bbox = track.tracker.update(img)
        track.ok = ok
        track.bbox = bbox
    
        print(bbox)
#(x1, y1), x2 - x1, y2 - y1
        cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), track.color, 2)

    cv2.imshow('img', img)
        
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


