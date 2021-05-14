from enum import Enum
import numpy as np
import matplotlib.pyplot as plt
import skimage.color
import utils
import mean_shift
from plotclusters3D import plotclusters3D


class FeatureType(Enum):
    """
    Specifies the type of image features - either 3D color features or 5D color and spatial feature.
    """
    color = 3             # uses only color space
    color_spatial = 5     # uses color and spatial information


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
    # show original image
    plt.imshow(img_rgb)
    plt.show()
    # preprocess image
    img_lab = skimage.color.rgb2lab(img_rgb)
    img_lab = utils.retrieve_features(img_lab, feature_type)
    # perform image segmentation using mean shift algorithm
    labels, peaks = mean_shift.ms_speedup2(img_lab, r, c)
    # plot clusters
    plotclusters3D(img_lab.T, labels, peaks)
    # postprocess segmentation data
    segments = dict(zip(np.unique(labels), np.flip(peaks, axis=0)))
    segmented = np.array([segments[l] if l in segments.keys() else l for l in labels])
    img_rgb_seg = skimage.color.lab2rgb(segmented.reshape(img_rgb.shape))
    # show segmented image
    plt.imshow(img_rgb_seg)
    plt.show()


def test_mean_shift(path):
    data = utils.load_test_data(path)
    print("data shape: ", data.shape)

    labels, peaks = mean_shift.ms_no_speedup(data, r=2)
    print("mean shift - # cluster: %s, peaks: %s" % (np.unique(labels), peaks))
    plotclusters3D(data.T, labels, peaks)

    labels, peaks = mean_shift.ms_speedup1(data, r=2)
    print("1. speedup - # cluster: %s, peaks: %s" % (np.unique(labels), peaks))
    plotclusters3D(data.T, labels, peaks)

    labels, peaks = mean_shift.ms_speedup2(data, r=2, c=4)
    print("2. speedup - # cluster: %s, peaks: %s" % (np.unique(labels), peaks))
    plotclusters3D(data.T, labels, peaks)


if __name__ == '__main__':
    # test_mean_shift(path="data/pts.mat")

    imgs = utils.load_images(dir='./img/', filenames=["deer10.png", "181091.jpg", "368078.jpg"])
    img_rgb = imgs[0]

    # Image segmentation
    r = 20
    c = 4
    feature_type = FeatureType.color  # color, color_spatial
    image_segmentation(img_rgb, r, c, feature_type)

    print("Finished, cheers!")

