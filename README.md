<p align="center">
    <img src="https://github.com/agusle/car-images-classification/blob/main/img/project-logo.png" width = 400 height = 400>
</p>

<p align="center">
    <a href="https://github.com/agusle/car-images-classification/commits/main">
    <img src="https://img.shields.io/github/last-commit/agusle/car-images-classification?logo=Github"
         alt="GitHub last commit">
    <a href="https://github.com/agusle/car-images-classification/issues">
    <img src="https://img.shields.io/github/issues-raw/agusle/car-images-classification?logo=Github"
         alt="GitHub issues">
    <a href="https://github.com/agusle/car-images-classification/pulls">
    <img src="https://img.shields.io/github/issues-pr-raw/agusle/car-images-classification?logo=Github"
         alt="GitHub pull requests">
</p>

<p align="center">
  <a href="#-about">About</a> • 
  <a href="#%EF%B8%8F-install-and-run">Install and Run</a> •
  <a href="#-usage">Usage</a> •
  <a href="#-contribute">Contribute</a> •
</p>

------------------

## 📖 About
- **Problem**: Car Image Classification for E-Commerce.

- **Industries**: e-commerce, insurtech, goverment many others related.

- **Solution**: Predict vehicle make and model from unstructured e-commerce images. Trained on a pre-built dataset of 196 classes. Visualized and cleaned the dataset, pre-processed and augmented data, and trained a fine-grained classification model using ResNet50 convolutional neural network achieving 56% accuracy in the prediction of make and model combined. Deployed in AWS instances using Docker, using an API based web-service application.

You can see detailed information in the following **reports**:
 - [Model Evaluation Report](https://github.com/agusle/car-images-classification/blob/main/reports/Evaluation_report.md)

------------------

## ⚡️ Install and Run 

Listed below you'll find an example or application usage to run the services using compose:

You can use `Docker` to easily install all the needed packages and libraries. Two Dockerfiles are provided for both CPU and GPU support.

- **CPU:**

```bash
$ docker build -t car-classification -f docker/Dockerfile .
```

- **GPU:**

```bash
$ docker build -t car-classification-f docker/Dockerfile_gpu .
```

### Run Docker

```bash
$ docker run --rm --net host -it \
    -v $(pwd):/home/app/src \
    --workdir /home/app/src \
    car-classification \
    bash
```

### Run Unit test


```bash
$ pytest tests/
```
------------------

## 👀 Usage

### 1. Prepare your data

As a first step, I must extract the images from the file `car_ims.tgz` and put them inside the `data/` folder. Also place the annotations file (`car_dataset_labels.csv`) in the same folder. It should look like this:

```
data/
    ├── car_dataset_labels.csv
    ├── car_ims
    │   ├── 000001.jpg
    │   ├── 000002.jpg
    │   ├── ...
```

Then, you should be able to run the script [scripts/prepare_train_test_dataset.py](https://github.com/agusle/car-images-classification/blob/main/scripts/prepare_train_test_dataset.py). It will format your data in a way Keras can use for training our CNN model.

### 2. Train the CNN (Resnet50)

After I have our images in place, it's time to create the CNN and train it on our dataset. To do so, I will make use of [scripts/train.py](https://github.com/agusle/car-images-classification/blob/main/scripts/train.py).

The only input argument it receives is a YAML file with all the experiment settings like dataset, model output folder, epochs, learning rate, data augmentation, etc.

Each time you are going to train a new a model, I recommend you to create a new folder inside the `experiments/` folder with the experiment name. Inside this new folder, create a `config.yml` with the experiment settings. I also encourage you to store the model weights and training logs inside the same experiment folder to avoid mixing things between different runs. The folder structure should look like this:

```bash
experiments/
    ├── exp_001
    │   ├── config.yml
    │   ├── logs
    │   ├── model.01-6.1625.h5
    │   ├── model.02-4.0577.h5
    │   ├── model.03-2.2476.h5
    │   ├── model.05-2.1945.h5
    │   └── model.06-2.0449.h5
    ├── exp_002
    │   ├── config.yml
    │   ├── logs
    │   ├── model.01-7.4214.h5
    ...
```
The script [scripts/train.py](https://github.com/agusle/car-images-classification/blob/main/scripts/train.py) is already coded but it makes use of external functions from other project modules like: 

- `utils.load_config()`: Takes as input the path to an experiment YAML configuration file, loads it and returns a dict.
- `resnet50.create_model()`: Returns a CNN ready for training or for evaluation, depending on the input parameters received. Part of coding this functions will require you to create the layers of your first CNN with Keras.
- `data_aug.create_data_aug_layer()`: Used by `resnet50.create_model()`. This function adds data augmentation layers to our model that will be used only while training.

### 3. Evaluate your trained model

After running many experiments and having a potentially good model trained. It's time to check its performance on our test dataset and prepare a nice report with some evaluation metrics.

I used the following notebooks:
-  [Model_Evaluation_1.ipynb](https://github.com/agusle/car-images-classification/blob/main/notebooks/Model_Evaluation_1.ipynb)
-  [Model_Evaluation_2.ipynb](https://github.com/agusle/car-images-classification/blob/main/notebooks/Model_Evaluation_2.ipynb)
-  [Model_Evaluation_3.ipynb](https://github.com/agusle/car-images-classification/blob/main/notebooks/Model_Evaluation_3.ipynb)

### 4. Improve classification by removing noisy background

As I already saw in the [notebooks/EDA.ipynb](https://github.com/agusle/car-images-classification/blob/main/notebooks/EDA.ipynb) file. Most of the images have a of background which may affect our model learning during the training process.

It's a good idea to remove this background. One thing I can do is to use a Vehicle detector to isolate the car from the rest of the content in the picture.

I will use [Detectron2](https://github.com/facebookresearch/detectron2) framework for this. It offers a lot of different models, you can check in its [Model ZOO](https://github.com/facebookresearch/detectron2/blob/main/MODEL_ZOO.md#faster-r-cnn). I will use for this assignment the model called "R101-FPN".

In particular, I will use a detector model trained on [COCO](https://cocodataset.org) dataset which has a good balance betIen accuracy and speed. This model can detect up to 80 different types of objects but here I're only interested on getting two out of those 80, those are the classes "car" and "truck".

For this assignment, it is used:

- [scripts/remove_background.py](https://github.com/agusle/car-images-classification/blob/main/scripts/remove_background.py): It will process the initial dataset used for training your model on **item (3)**, removing the background from pictures and storing the resulting images on a new folder.
- [utils/detection.py](https://github.com/agusle/car-images-classification/blob/main/utils/detection.py): This module loads our detector and implements the logic to get the vehicle coordinate from the image.

Now you have the new dataset in place, it's time to start training a new model and checking the results in the same way as I did for steps items **(3)** and **(4)**.

------------------

## 👍 Contribute
**Please follow these steps to get your work merged in.**

1. Add a [GitHub Star](https://github.com/agusle/car-images-classification) to the project.
2. Clone repo and create a new branch: `$ git checkout https://github.com/agusle/car-images-classification -b name_for_new_branch.`
3. Add a feature, fix a bug, or refactor some code :)
4. Write/update tests for the changes you made, if necessary.
5. Update `README.md`, if necessary.
4. Submit Pull Request with comprehensive description of changes
