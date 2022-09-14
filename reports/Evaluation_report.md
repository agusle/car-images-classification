                                                                                                    Agustín Leperini
                                                                                                    25.06.22
---
# Model Evaluation Report
## Introduction 

In this evaluation report I'm explaining the results of several experiments carried out to use a pre-trained well known convolutional neural network called [ResNet50](https://www.kaggle.com/datasets/keras/resnet50) to classify vehicules images by:
- Transfer Learning
- Fine-Tuning

During this iterative process using AWS Elastic Computing (EC2) cloud server I evaluate the trade-off between the resources and time consumption with the accuracy we desire, to choose the best approach between the mentioned techniques.

## Hardware specifications of server used for training:
### GPU:
NVIDIA-SMI 470.129.06  
Driver Version: 470.129.06  
CUDA Version: 11.4 
Model: Tesla K80 
Memory: 11.441 MiB 

*Running the Resnet_50 training took in all cases about 10.000 MiB and the percentage of gpu used was around 90%.*

## Dataset
### Exploratory Data Analysis:
Model tranining was based on a vehicle dataset of 16,185 images divided in 196 different classes, corresponding 8144 to train (50,3%) and 8041 to test (49,7%). Can see more details of dataset on the following notebook: [EDA - Notebook](https://github.com/anyoneai/sprint5-project/blob/AgustinLeperini_assignment/notebooks/EDA.ipynb)

### Dataset preprocessing:
Later on the project a background removal was applied image by image with an object detection algoritm called Faster R-CNN (specially faster_rcnn_R_101_FPN_3x) from [Detectron2](https://ai.facebook.com/tools/detectron2/) platform to identify vehicles area and crop the image in order to have a more accurate input.

*Running this process took about 2:30hs, 2.000 MiB and the percentage of gpu used was around 85%.*

## Experiments
### Playing with differents parameters to reach best performance:
The following table shows all the experiments, their parameters and the corresponding results: 

|  # exp | Technique |                                         Regularization                                        |                                                                              Modifications from previous experiment                                                                             | Epochs | Train Accuracy | Validation Accuracy | Test Accuracy |       Evaluation       |
|:------:|:---------:|:---------------------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:------:|:--------------:|:-------------------:|:-------------:|:----------------------:|
| 1      | TL        | 1-Data Aug: Flip,Rotation,Zoom,Translation   2-Dropout (0.20)                                 |                                                                                                                                                                                                 |   25   |     0.4458     |        0.2475       |   Not tested  |       Not tested       |
| 2      | TL        | 1-Data Aug: Flip,Rotation,Zoom,Translation   2-Dropout (0.20)                                 | Increase batch size (64) Increase learning rate (0.001)                                                                                                                                         |   50   |     0.5247     |        0.2801       |   Not tested  |       Not tested       |
| 3      | TL        | 1-Data Aug: Flip,Rotation,Zoom,Translation   2-Dropout (0.20)  3-Kernel: ElasticNet (l1_l2)   | Decrease learning rate (0.0001) Add Elastic net regularization                                                                                                                                  |   75   |     0.3892     |        0.2359       |   Not tested  |       Not tested       |
| 4      | TL        | 1-Data Aug: Flip,Rotation,Zoom,Translation   2-Dropout (0.40)  3-Kernel: ElasticNet (l1_l2)   | Increase learning rate (0.01) Increase dropout (0.40)                                                                                                                                           |   50   |     0.3874     |        0.2070       |   Not tested  |       Not tested       |
| 5      | FT        | 1-Data Aug: Flip,Rotation,Zoom,Translation   2-Dropout (0.40)  3-Kernel: ElasticNet (l1_l2)   | Increase learning rate (0.001) Increase dropout (0.40) Increase rotation-zoom-translation (0.5) Decrease batch size (32)                                                                        |   75   |     0.1940     |        0.1929       |   Not tested  |       Not tested       |
| 6      | FT        | 1-Data Aug: Flip,Rotation,Zoom,Translation   2-Dropout (0.50)  3-Kernel: ElasticNet (l1_l2)   | Increase dropout (0.50) Decrease batch size (32)                                                                                                                                                |   25   |     0.1912     |        0.1812       |   Not tested  |       Not tested       |
| 8      | FT        | 1-Data Aug: Flip,Rotation,Zoom,Translation   2-Dropout (0.50)  3-Kernel: ElasticNet (l1_l2)   | Decrease learning rate (0.0001) Flip = horizontal                                                                                                                                               |   25   |     0.2831     |        0.2334       |     0.2581    |   [Model Evaluation 1 ](https://github.com/anyoneai/sprint5-project/blob/AgustinLeperini_assignment/notebooks/Model_Evaluation_1.ipynb)  |
| 9      | FT        | 1-Data Aug: Flip,Rotation,Zoom,Contrast   2-Dropout (0.50)  3-Kernel: l2                      | Increase batch size (64) Increase learning rate (0.0009) Add Early stopping Change Elastic Net for l2 regularization Change translation by contrast Decrease flip-rotation-zoom-contrast (0.25) |   33   |     0.7215     |        0.5514       |   Not tested  |       Not tested       |
| 10     | FT        | 1-Data Aug: Flip,Rotation,Zoom,Contrast   2-Dropout (0.50)  3-Kernel: l2                      | Remove Early stopping Decrease flip-rotation-zoom-contrast (0.15)                                                                                                                               |   150  |     0.9827     |        0.7328       |     0.5575    |   [Model Evaluation 2](https://github.com/anyoneai/sprint5-project/blob/AgustinLeperini_assignment/notebooks/Model_Evaluation_2.ipynb)  |
| **11** | **FT**    | **1-Data Aug: Flip,Rotation,Zoom,Contrast   2-Dropout (0.50)  3-Kernel: l2**                  | **Decrease learning rate (0.00009)**                                                                                                                                                            | **50** |   **0.9912**   |      **0.8132**     |   **0.5665**  | [**Model Evaluation 3**](https://github.com/anyoneai/sprint5-project/blob/AgustinLeperini_assignment/notebooks/Model_Evaluation_3.ipynb) |


- TL = Transfer Learning
- FT = Fine-Tuning

## Conclusion

As expected, some optimization techniques worked on data while others did not. I could increase test accuracy from **25.81% to 56.54%**. 

The best results come from **experiment #11 where validation accuracy reach 81.32%, and a minimum validation loss 0.8807**.
To report the accuracy and loss, I used [Tensorboard](https://www.tensorflow.org/tensorboard?hl=es-419). It can track and visualize loss and accuracy metrics during training and after it. The possibility to see these variables "on live" helps to identify some problems early.

![epoch_accuracy_exp_11](https://github.com/agusle/car-images-classification/blob/main/notebooks/epoch_accuracy_exp_11.jpg)
![epoch_accuracy_exp_11](https://github.com/agusle/car-images-classification/blob/main/notebooks/epoch_loss_exp_11.jpg)

During the evaluation of all experiments I can conclude that the following data, model and compile configuration helped train, validation and test accuracy reach and acceptable level: 

- Turning from Transfer learning to Fine-Tuning training unfreezeing all of the base model and retrain the whole model end-to-end.
- Decreasing learning rate and increase training epochs.
- Decreasing data augmentation rate.
- Replacing translation data augmentation layer by contrast data augmentation.
- Increase batch size from 32 to 64.
- L2 kernel regularization on dense layer performs better than Elastic Net (l1_l2).
- Early stopping is not recommendable because validation accuracy quite improves besides a probable overfitting.
- Cropping images background improves train and validation accuracy significantly.

## What can be improved
Dataset
- Obtain more images to dataset it's always useful.
- Optimize training data with "K-Fold Cross-Validation" to divide the images into K parts of equal size and then train the model K number of times with a different training and validation set.

Model 
- Try with different data augmentation layer (e.g adding Random Brightness)
- Increase kernel regularization rate on dense model layer.
- Try adding kernel regularization on more layers.
- Do ensemble learning combining the predictions from multiple models.

Model compilation
- Tune epsilon hyperparameter in Adam Optimizer.
