"""
This script will be used to remove noisy background from cars images to
improve the quality of our data and get a better model.
The main idea is to use a vehicle detector to extract the car
from the picture, getting rid of all the background, which may cause
confusion to our CNN model.
We must create a new folder to store this new dataset, following exactly the
same directory structure with its subfolders but with new images.
"""

import argparse
from utils.utils import walkdir
import os
from tensorflow import keras
from utils.detection import get_vehicle_coordinates
import cv2


def parse_args():
    parser = argparse.ArgumentParser(description="Train your model.")
    parser.add_argument(
        "data_folder",
        type=str,
        help=(
            "Full path to the directory having all the cars images. Already "
            "splitted in train/test sets. E.g. "
            "`/home/app/src/data/car_ims_v1/`."
        ),
    )
    parser.add_argument(
        "output_data_folder",
        type=str,
        help=(
            "Full path to the directory in which we will store the resulting "
            "cropped pictures. E.g. `/home/app/src/data/car_ims_v2/`."
        ),
    )

    args = parser.parse_args()

    return args


def main(data_folder, output_data_folder):
    """
    Parameters
    ----------
    data_folder : str
        Full path to train/test images folder.

    output_data_folder : str
        Full path to the directory in which we will store the resulting
        cropped images.
    """
    # check if the directory exist in destination directory
    if not os.path.exists(output_data_folder):
        os.makedirs(output_data_folder)

    # check if train and test directories already exists, otherwise create it
    train_path = os.path.join(output_data_folder, "train")
    if not os.path.exists(train_path):
        os.makedirs(train_path)
    #
    test_path = os.path.join(output_data_folder, "test")
    if not os.path.exists(test_path):
        os.makedirs(test_path)

    # For this function, you must:
    #   1. Iterate over each image in `data_folder`, you can
    #      use Python `os.walk()` or `utils.waldir()``
    counter = 0
    for dirpath, filename in walkdir(data_folder):

        #   2. Load the image
        image_path = os.path.join(dirpath, filename)
        image = cv2.imread(image_path)

        #   3. Run the detector and get the vehicle coordinates, use
        #      utils.detection.get_vehicle_coordinates() for this task
        coord = get_vehicle_coordinates(image)
        image_cropped = image[coord[1] : coord[3], coord[0] : coord[2]]
        label = dirpath.split("/")[4]
        dataset_type = dirpath.split("/")[3]

        #   4. Extract the car from the image and store it in
        #      `output_data_folder` with the same image name. You may also need
        #      to create additional subfolders following the original
        #      `data_folder` structure.

        if dataset_type == "train":
            label_train_path = os.path.join(train_path, label)
            if not os.path.exists(label_train_path):
                os.makedirs(label_train_path)
            image_cropped_path = os.path.join(label_train_path, filename)
            counter += 1

        if dataset_type == "test":
            label_test_path = os.path.join(test_path, label)
            if not os.path.exists(label_test_path):
                os.makedirs(label_test_path)
            image_cropped_path = os.path.join(label_test_path, filename)
            counter += 1

        # counter every 500 images cropped
        if counter % 50 == 0:
            print(f"Image:{counter} - {dataset_type} - {label}")

        cv2.imwrite(image_cropped_path, image_cropped)


if __name__ == "__main__":
    args = parse_args()
    main(args.data_folder, args.output_data_folder)
