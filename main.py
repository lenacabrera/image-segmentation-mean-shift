import utils
import mean_shift
from plotclusters3D import plotclusters3D
import os
import scipy.io
import matplotlib.pyplot as plt
from enum import Enum
import numpy as np
from scipy.spatial import distance
import skimage.color


class FeatureType(Enum):
    color = 0             # uses only color space
    color_spatial = 1     # uses color and spatial information


def image_segmentation(img_rgb, r, c, feature_type):
    # cluster the image data in CIELAB color space by first converting the RGB color vectors to CIELAB using
    # color.rgb2lab(img)
    # then convert the resulting cluster centers back to RGB using color.lab2rgb()
    """


    Parameters
    ----------
    img : input color RGB image
    r : radius

    Returns
    -------

    """
    img_lab = skimage.color.rgb2lab(img_rgb)
    img_lab = utils.retrieve_features(img_lab, FeatureType.color_spatial)
    labels, peaks = mean_shift.mean_shift_speedup2(img_lab, r=2, c=4)

    print(np.unique(labels))
    plotclusters3D(img_lab.T, labels, peaks)

    # utils.retrieve_cluster_centers()
    pass


def test_mean_shift(path):
    data = utils.load_test_data(path)
    print("data shape: ", data.shape)

    labels, peaks = mean_shift.mean_shift_new(data, r=2)
    print("mean shift - # cluster: ", np.unique(labels))
    plotclusters3D(data.T, labels, peaks)

    # labels, peaks = mean_shift.mean_shift(data, r=2)
    # print("mean shift - # cluster: ", np.unique(labels))
    # plotclusters3D(data.T, labels, peaks)
    #
    # labels, peaks = mean_shift.mean_shift_speedup1(data, r=2)
    # print("1. speedup - # cluster: ", np.unique(labels))
    # plotclusters3D(data.T, labels, peaks)

    # labels, peaks = mean_shift.mean_shift_speedup2(data, r=2, c=4)
    # print("2. speedup - # cluster: ", np.unique(labels))
    # plotclusters3D(data.T, labels, peaks)


if __name__ == '__main__':
    test_mean_shift(path="data/pts.mat")

    imgs = utils.load_images(dir='./img/', filenames=["deer10.png", "181091.jpg", "368078.jpg"])
    img_rgb = imgs[0]

    # Image segmentation
    r = 2
    c = 4
    feature_type = FeatureType.color  # color, color_spatial
    # image_segmentation(img_rgb, r, c, feature_type)

    print("Finished, cheers!")

