import os
import sys
import random
import cv2
import random

# Root directory of the project
ROOT_DIR = os.path.abspath("./")
print(os.listdir(ROOT_DIR))
# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
import mrcnn.model as modellib

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

    # Skip detections with < 80% confidence
    DETECTION_MIN_CONFIDENCE = 0.80

class InferenceConfig(TomatoConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

def load_model():
    inference_config = InferenceConfig()

    # Recreate the model in inference mode
    model = modellib.MaskRCNN(mode="inference", 
                          config=inference_config,
                          model_dir=MODEL_DIR)

    # Get path to saved weights
    # Either set a specific path or find last trained weights
    model_path = os.path.join("/Users/a104133/Projects/Tomato_detection/logs/", "mask_rcnn_tomato.h5")

    # Load trained weights
    print("Loading weights from ", model_path)
    model.load_weights(model_path, by_name=True)

    print("Loaded weights")
    return model

cap = cv2.VideoCapture('/Users/a104133/Downloads/TEST3.mov')

class Track:
    bbox = ()
    tracker = None
    ok = None
    color = ()

tracks = []
init_tracks = True

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    if init_tracks:
        init_tracks = False

        model = load_model()

        results = model.detect([frame], verbose=1)
        
        # Create tracks fro all detected objects
        for rois in results[0]['rois']:
            y1, x1, y2, x2 = rois
            track = Track()
            track.bbox = (x1, y1, x2, y2)
            track.tracker = cv2.TrackerKCF_create()
            track.ok = None
            track.color = (random.randint(10,250), random.randint(10,250), random.randint(10,250))
            tracks.append(track)
        
    for track in tracks:
        if track.ok is None:
            track.ok = track.tracker.init(frame, track.bbox)

        ok, bbox = track.tracker.update(frame)
        track.ok = ok
        track.bbox = bbox
    
        print(bbox)
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), track.color, 2)

    cv2.imshow('img', frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()