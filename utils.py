import scipy.io
import scipy.ndimage
import matplotlib.pyplot as plt
import numpy as np
from main import FeatureType, Image


def load_test_data(path="data/pts.mat"):
    """
    Loads test data which stores a set of 3D points belonging to two 3D Gaussians.

    Parameters
    ----------
    path : path to data file

    Returns
    -------
    data : data with shape [3]x[2000]

    """
    data = scipy.io.loadmat(path)["data"]
    return data


def load_images(filenames, dir='./img/'):
    """
    Loads images specified by their filenames

    Parameters
    ----------
    filenames : list of names of image files
    dir : image directory

    Returns
    -------
    images : list of images as arrays

    """
    images = []
    for filename in filenames:
        images.append(plt.imread(dir + filename, 0))
    return images


def load_image(image):
    return plt.imread(image.value['src'], 0)


def retrieve_features(img, feature_type):
    """
    Preprocesses image data and extracts features for each pixel. One of two types of features is used:
    (1) CIELAB color space, with 3 color channels (3D feature vector)
    (2) CIELAB color space + spatial information, with 3 color channels and 2 coordinates (5D feature vector)

    Returns
    -------
    image : extracted image features as array with shape [3]x[#pixels] or [5]x[#pixels]
    """
    if feature_type.value == FeatureType.color.value:
        image = img.reshape(img.shape[0] * img.shape[1], img.shape[2])
    if feature_type.value == FeatureType.color_spatial.value:
        x = np.array(list(range(img.shape[0])))
        y = np.array(list(range(img.shape[1])))
        coordinates = np.array(np.meshgrid(x, y))
        image = np.append(img, coordinates.T, axis=2)
        image = image.reshape(image.shape[0] * image.shape[1], image.shape[2])
    return image.T


def apply_filter(img_rgb, type):
    if type == 'gaussian':
        return scipy.ndimage.gaussian_filter(img_rgb, sigma=1)
    if type == 'median':
        return scipy.ndimage.median_filter(img_rgb, size=1)

