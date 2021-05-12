import utils
import mean_shift
import os
import scipy.io
import matplotlib.pyplot as plt
from enum import Enum
import numpy as np
from scipy.spatial import distance
import skimage.color
from main import Features


def load_data(path):
    """ Loads the data """
    data = scipy.io.loadmat(path)["data"]
    return data


def load_images(dir, filenames):
    print("Load image ...")
    images = []
    for filename in filenames:
        images.append(plt.imread(dir + filename, 0))
    return images


def preprocess_img(img, features):
    """
    - create 3-by-p matrix in case you use color features, where p is the number of pixels in the input image
    - if we want to include spatial position information as well, define the feature vector as a 5D vector specifying
      the color channels and x, y coordinates of each pixel
    - use python functions to convert between color spaces
    - avoid using loops whenever possible!

    Returns
    -------

    """
    img = reshape_img(img, features)
    return img


def reshape_img(img, features):

    if features.value == Features.color.value:
        image = img.reshape((img.shape[2], img.shape[0] * img.shape[1]))
    if features.value == Features.color_spatial.value:
        x = np.array(list(range(img.shape[0])))
        y = np.array(list(range(img.shape[1])))
        coordinates = np.array(np.meshgrid(x, y))
        image = np.append(img, coordinates.T, axis=2)
        image = image.reshape((image.shape[2], image.shape[0] * image.shape[1]))
    return image