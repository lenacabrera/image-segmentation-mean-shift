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


def image_segmentation(img, r, c, feature_type):
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
    data = utils.load_data("data/pts.mat")
    labels, peaks = mean_shift.mean_shift(data, r=2)
    # labels, peaks = mean_shift.mean_shift_opt2(data, r=2, c=4)
    plotclusters3D(data, labels.T, peaks)

    print(data.shape)
    imgs = utils.load_images(dir='./img/', filenames=["deer10.png", "181091.jpg", "368078.jpg"])
    img_rgb = imgs[0]
    img_lab = skimage.color.rgb2lab(img_rgb)
    img_lab = utils.preprocess_img(img_lab, FeatureType.color_spatial)

    # print(img.shape, img.T[0])
    # distances = compute_distances(img[:, 0], img)
    # print(distances)
    # print(find_peak(img_lab, 0, 7))

    # labels, peaks = mean_shift.mean_shift(data, r=2)
    labels, peaks = mean_shift.mean_shift_opt2(data, r=2, c=4)
    print("Finished, cheers!")

    # Image segmentation
    r = 30
    c = 4.5
    feature_type = FeatureType.color_spatial
    image_segmentation(img_rgb, r, c, feature_type)

