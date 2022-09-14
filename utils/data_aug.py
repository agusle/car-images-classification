from tensorflow import keras


def create_data_aug_layer(data_aug_layer):
    """
    Use this function to parse the data augmentation methods for the
    experiment and create the corresponding layers.

    It will be mandatory to support at least the following three data
    augmentation methods (you can add more if you want):
        - `random_flip`: keras.layers.RandomFlip()
        - `random_rotation`: keras.layers.RandomRotation()
        - `random_zoom`: keras.layers.RandomZoom()

    See https://tensorflow.org/tutorials/images/data_augmentation.

    Parameters
    ----------
    data_aug_layer : dict
        Data augmentation settings coming from the experiment YAML config
        file.

    Returns
    -------
    data_augmentation : keras.Sequential
        Sequential model having the data augmentation layers inside.
    """
    # Parse config and create layers
    # You can use as a guide on how to pass config parameters to keras
    # looking at the code in `scripts/train.py`

    data_aug_tech = {
        "random_flip": keras.layers.RandomFlip,
        "random_rotation": keras.layers.RandomRotation,
        "random_zoom": keras.layers.RandomZoom,
        "random_contrast": keras.layers.RandomContrast,
    }

    data_aug_layers = []
    if data_aug_layer != None:  # check to ignore empty dictionary
        for data_aug_type, data_aug_arg in data_aug_layer.items():
            if (
                data_aug_type in data_aug_layer.keys()
            ):  # check if technique exists in input config dictionary
                data_aug_layers.append(
                    data_aug_tech[data_aug_type](**data_aug_arg)
                )

    # Return a keras.Sequential model having the the new layers created
    # Assign to `data_augmentation` variable
    data_augmentation = keras.Sequential(data_aug_layers)

    return data_augmentation
