import os
import scipy.io
import matplotlib.pyplot as plt
from enum import Enum
import numpy as np


class Features(Enum):
    color = 0
    color_spatial = 1



def load_data(path):
    """ Loads the data """
    data = scipy.io.loadmat(path)["data"]
    return data


def load_images(dir, filenames):
    images = []
    for filename in filenames:
        images.append(plt.imread(dir + filename, 0))
    return images


def reshape_img(img, features):

    if features == Features.color:
        image = img.reshape((img.shape[2], img.shape[0] * img.shape[1]))
    if features == Features.color_spatial:
        x = np.array(list(range(img.shape[0])))
        y = np.array(list(range(img.shape[1])))
        coordinates = np.array(np.meshgrid(x, y))
        image = np.append(img, coordinates.T, axis=2)
        image = image.reshape((image.shape[2], image.shape[0] * image.shape[1]))
    return image


def find_peak(data, idx, r):
    """
    Compute density peak for point p whose column index is provided.

    Parameters
    ----------
    data : n-dimensional dataset
    idx : column index of data point
    r : search window radius

    Returns
    -------

    """
    pass


def mean_shift(data, r):
    """
    Computes peaks and corresponding labels of all points in the data.

    Parameters
    ----------
    data : n-dimensional dataset
    r : search window radius

    Returns
    -------

    """
    labels = []
    peaks = []

    # call find_peak for each point and then assign a label to each point according to its peak
    # after each call compare peaks and merge similar ones (two peaks are the same if their distance is < r/2)
    # if found peak already exists in peaks, computed peak is discarded and is given label associated with peak in peaks

    return labels, peaks


def mean_shift_opt(data, r, c):
    """
    First speedup: "Basin of attraction"
    Associates each data point that is at a distance <= r from the peak with the cluster defined by that peak.

    Parameters
    ----------
    data : n-dimensional dataset
    r : search window radius
    c : constant value

    Returns
    -------

    """


def find_peak_opt(data, idx, r, threshold, c=4):
    """
    Second speedup:
    Associates points that are within a distance of r/c of the search path with the converged peak

    Parameters
    ----------
    data : n-dimensional dataset
    idx : column index of data point
    r : search window radius
    threshold :
    c : constant value

    Returns
    -------
    peaks :
    cpts : vector storing 1 for each point that is a distance of r/c from the path and 0 otherwise

    """


def preprocessing():
    """
    - create 3-by-p matrix in case you use color features, where p is the number of pixels in the input image
    - if we want to include spatial position information as well, define the feature vector as a 5D vector specifying
      the color channels and x, y coordinates of each pixel
    - use python functions to convert between color spaces
    - avoid using loops whenever possible!

    Returns
    -------

    """
    pass


def compute_distances(data_point, data, metric='euclidean'):
    """
    Computes the distances between a data point and all other data points.

    Parameters
    ----------
    data_point :
    data :
    metric :

    Returns
    -------

    """
    pass


def image_segmentation(img, r):
    """


    Parameters
    ----------
    img : input color RGB image
    r : radius

    Returns
    -------

    """

    # cluster the image data in CIELAB color space by first converting the RGB color vectors to CIELAB using
    # color.rgb2lab(img)
    # then convert the resulting cluster centers back to RGB using color.lab2rgb()
    pass


if __name__ == '__main__':
    # data = load_data("data/pts.mat")
    images = load_images(dir='./img/', filenames=["181091.jpg", "368078.jpg"])
    img = images[0]
    img = reshape_img(img, Features.color_spatial)
    print(img.shape, img[0])

