import os
import scipy.io
import matplotlib.pyplot as plt
from enum import Enum
import numpy as np
from scipy.spatial import distance
import skimage.color


class Features(Enum):
    color = 0             # uses only color space
    color_spatial = 1     # uses color and spatial information


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
    labels, peaks = mean_shift(img_lab, 10)
    print("Finished, cheers!")

