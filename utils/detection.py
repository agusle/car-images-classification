# Load here your Detection model
# The chosen detector model is "COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"
# because this particular model has a good balance between accuracy and speed.
# You can check the following Colab notebook with examples on how to run
# Detectron2 models
# https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5.

import detectron2
from detectron2.utils.logger import setup_logger
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
import torch

# Get a copy of the default config: CfgNode instance and
cfg = get_cfg()

# Run on CPU
# cfg.MODEL.DEVICE = "cpu"

# add  specifics config
cfg.merge_from_file(
    model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")
)
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5

# load model pre-trained weights from chosen detectron model
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
    "COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"
)

# Assign the loaded detection model to global variable DET_MODEL
DET_MODEL = DefaultPredictor(cfg)


def get_vehicle_coordinates(img):
    """
    This function will run an object detector over the the image, get
    the vehicle position in the picture and return it.

    Many things should be taken into account to make it work:
        1. Current model being used can detect up to 80 different objects,
           we're only looking for 'cars' or 'trucks', so you should ignore
           other detected objects.
        2. The object detector may find more than one vehicle in the picture,
           you must then, choose the one with the largest area in the image.
        3. The model can also fail and detect zero objects in the picture,
           in that case, you should return coordinates that cover the full
           image, i.e. [0, 0, width, height].
        4. Coordinates values must be integers, we're making reference to
           a position in a numpy.array, we can't use float values.

    Parameters
    ----------
    img : numpy.ndarray
        Image in RGB format.

    Returns
    -------
    box_coordinates : tuple
        Tuple having bounding box coordinates as (left, top, right, bottom).
        Also known as (x1, y1, x2, y2).
    """
    outputs = DET_MODEL(img)

    classes = outputs["instances"].pred_classes

    # look for car(2) or trucks(7) in output classes
    if 2 in classes or 7 in classes:
        # filter by car or track outputs
        car_truck = outputs["instances"][(classes == 2) | (classes == 7)]
        # transform boxes object to area
        car_truck_area = car_truck.pred_boxes.area()
        # get tensor item with max area
        car_truck_area = torch.argmax(car_truck_area).item()

        # transform max area into a box coordenates of integers
        box_coordinates = (
            car_truck.pred_boxes.tensor[car_truck_area].to(torch.int).tolist()
        )
    else:
        # original image
        image_heigth = img.shape[0]
        image_width = img.shape[1]
        box_coordinates = [0, 0, image_width, image_heigth]

    return box_coordinates
