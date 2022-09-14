"""
This script will be used to separate and copy images coming from
`car_ims.tgz` (extract the .tgz content first) between `train` and `test`
folders according to the column `subset` from `car_dataset_labels.csv`.
It will also create all the needed subfolders inside `train`/`test` in order
to copy each image to the folder corresponding to its class.

The resulting directory structure should look like this:
    data/
    ├── car_dataset_labels.csv
    ├── car_ims
    │   ├── 000001.jpg
    │   ├── 000002.jpg
    │   ├── ...
    ├── car_ims_v1
    │   ├── test
    │   │   ├── AM General Hummer SUV 2000
    │   │   │   ├── 000046.jpg
    │   │   │   ├── 000047.jpg
    │   │   │   ├── ...
    │   │   ├── Acura Integra Type R 2001
    │   │   │   ├── 000450.jpg
    │   │   │   ├── 000451.jpg
    │   │   │   ├── ...
    │   ├── train
    │   │   ├── AM General Hummer SUV 2000
    │   │   │   ├── 000001.jpg
    │   │   │   ├── 000002.jpg
    │   │   │   ├── ...
    │   │   ├── Acura Integra Type R 2001
    │   │   │   ├── 000405.jpg
    │   │   │   ├── 000406.jpg
    │   │   │   ├── ...
"""
import argparse
import pandas as pd
import os


def parse_args():
    parser = argparse.ArgumentParser(description="Train your model.")
    parser.add_argument(
        "data_folder",
        type=str,
        help=(
            "Full path to the directory having all the cars images. E.g. "
            "`/home/app/src/data/car_ims/`."
        ),
    )
    parser.add_argument(
        "labels",
        type=str,
        help=(
            "Full path to the CSV file with data labels. E.g. "
            "`/home/app/src/data/car_dataset_labels.csv`."
        ),
    )
    parser.add_argument(
        "output_data_folder",
        type=str,
        help=(
            "Full path to the directory in which we will store the resulting "
            "train/test splits. E.g. `/home/app/src/data/car_ims_v1/`."
        ),
    )

    args = parser.parse_args()

    return args


def main(data_folder, labels, output_data_folder):
    """
    Parameters
    ----------
    data_folder : str
        Full path to raw images folder.

    labels : str
        Full path to CSV file with data annotations.

    output_data_folder : str
        Full path to the directory in which we will store the resulting
        train/test splits.
    """

    #   1. Load labels CSV file
    labels_df = pd.read_csv(labels)
    # check if the directory exist in destination directory
    if not os.path.exists(output_data_folder):
        os.makedirs(output_data_folder)

    # check if train and test directories already exists, otherwise create it
    train_path = os.path.join(output_data_folder, "train")
    if not os.path.exists(train_path):
        os.makedirs(train_path)

    test_path = os.path.join(output_data_folder, "test")
    if not os.path.exists(test_path):
        os.makedirs(test_path)

    #   2. Iterate over each row in the CSV, create the corresponding
    #      train/test and class folders
    class_train_count = 0
    class_test_count = 0
    img_train_count = 0
    img_test_count = 0

    for index, row in labels_df.iterrows():
        if row["subset"] == "train":

            # check if class directory in train already exists, otherwise create it
            class_train_path = os.path.join(train_path, row["class"])
            if not os.path.exists(class_train_path):
                os.makedirs(class_train_path)
                class_train_count += 1

            # 3. Copy the image to the new folder structure. We recommend you to
            # use `os.link()` to avoid wasting disk space with duplicated files

            # check if image file already exists in destination,
            # otherwise link test images to destination path
            or_img_train_path = os.path.join(data_folder, row["img_name"])
            des_img_train_path = os.path.join(
                class_train_path, row["img_name"]
            )
            if not os.path.exists(des_img_train_path):
                os.link(or_img_train_path, des_img_train_path)
                img_train_count += 1

        elif row["subset"] == "test":

            # check if class directory in test already exists, otherwise create it
            class_test_path = os.path.join(test_path, row["class"])
            if not os.path.exists(class_test_path):
                os.makedirs(class_test_path)
                class_test_count += 1
            # 3. Copy the image to the new folder structure. We recommend you to
            # use `os.link()` to avoid wasting disk space with duplicated files

            # check if image file already exists in destination,
            # otherwise link test images to destination path
            or_img_test_path = os.path.join(data_folder, row["img_name"])
            des_img_test_path = os.path.join(class_test_path, row["img_name"])
            if not os.path.exists(des_img_test_path):
                os.link(or_img_test_path, des_img_test_path)
                img_test_count += 1

    return print(
        f"Dataset preparing completed:\n"
        f" Total images: {labels_df.shape[0]:,}\n"
        f" Train classes created: {class_train_count:,}\n"
        f" Test classes created: {class_test_count:,}\n"
        f" Train Images linked: {img_train_count:,}\n"
        f" Test Images Linked: {img_test_count:,}"
    )


if __name__ == "__main__":
    args = parse_args()
    main(args.data_folder, args.labels, args.output_data_folder)
