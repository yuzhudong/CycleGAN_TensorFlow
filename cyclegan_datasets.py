"""Contains the standard train/test splits for the cyclegan data."""

"""The size of each dataset. Usually it is the maximum number of images from
each domain."""
DATASET_TO_SIZES = {
    'adult2child_train_flatten': 50,
    'adult2child_train_flatten': 140
}

"""The image types of each dataset. Currently only supports .jpg or .png"""
DATASET_TO_IMAGETYPE = {
    'adult2child_train_flatten': '.csv',
    'adult2child_train_flatten': '.csv',
}

"""The path to the output csv file."""
PATH_TO_CSV = {
    'adult2child_train_flatten': './CycleGAN_TensorFlow/input/adult2child/adult2child_train_flatten.csv',
    'adult2child_train_flatten': './CycleGAN_TensorFlow/input/adult2child/adult2child_train_flatten.csv',
}
