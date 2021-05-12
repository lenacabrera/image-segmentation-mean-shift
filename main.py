import os
import scipy.io
import matplotlib.pyplot as plt
from enum import Enum
import numpy as np
from scipy.spatial import distance
import skimage.color


class Features(Enum):
    color = 0
    color_spatial = 1


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


def preprocess_data(img, features):
    img = reshape_img(img, features)
    return img


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


def find_peak(data, idx, r, threshold=0.01):
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

    # retrieve data point
    data_point = data[:, idx]

    peak = data_point
    old_peak = peak
    difference_peaks = np.ones((5,))
    similar_peaks = np.zeros((data.shape[1], 1))

    while np.all(difference_peaks > threshold):  # TODO all or any
        # define window
        distances = compute_distances(data_point, data)
        # check if peaks are similar, i.e. distance between them is smaller r/2
        if np.all(distances < r/2):
            are_similar_peaks = distances < r/2
            similar_peaks = are_similar_peaks.astype(int)

        # retrieve points in window
        window_indices = np.argwhere(distances <= r).flatten()
        neighbors = data.T[window_indices]
        # compute peak of window
        peak = np.mean(neighbors, axis=0)
        # compare peak to previous peak
        difference_peaks = np.abs(peak - old_peak)

    return peak, similar_peaks


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
    peaks = np.zeros((data.shape[1], data.shape[0]))
    labels = np.zeros((data.shape[1], ))
    unique_labels = 0

    for idx in range(data.shape[1]):
        peak, merge_peaks = find_peak(data, idx, r, threshold=0.01)
        # similar peaks get assigned the same label
        # same_peak = np.where(peaks == peak)
        # if len(same_peak) > 0:
        #     label = labels[0, same_peak[0]]
        # else:
        #     label = unique_labels
        #     unique_labels += 1
        #
        # labels[idx, 0] = label
        peaks[idx] = peak

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
    distances = np.array([scipy.spatial.distance.pdist(np.vstack((data_point, p)), metric=metric) for p in data.T])
    return distances


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
    images = load_images(dir='./img/', filenames=["deer10.png", "181091.jpg", "368078.jpg"])
    img_rgb = images[0]
    img_lab = skimage.color.rgb2lab(img_rgb)
    img_lab = preprocess_data(img_lab, Features.color_spatial)

    # print(img.shape, img.T[0])
    # distances = compute_distances(img[:, 0], img)
    # print(distances)
    # print(find_peak(img_lab, 0, 7))
    mean_shift(img_lab, 2)
    print("Finished, cheers!")

