from enum import Enum
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2lab, lab2rgb
import utils
import mean_shift
from plotclusters3D import plotclusters3D

class Image(Enum):

    img1 = {'src': "img/img1.jpg", 'dest':  "results/img1/"}
    img2 = {'src': "img/img2.jpg", 'dest':  "results/img2/"}
    img3 = {'src': "img/img3.jpg", 'dest':  "results/img3/"}
    img4 = {'src': "img/deer10.jpg", 'dest':  "results/img4/"}


class FeatureType(Enum):
    """
    Specifies the type of image features - either 3D color features or 5D color and spatial feature.
    """
    color = 3             # uses only color space
    color_spatial = 5     # uses color and spatial information


def image_segmentation(img_rgb, r, c, feature_type):
    """
    Performs image segmentation using mean-shift algorithm, i.e. partitioning an image into multiple segments of similar
    pixels. Converts an RGB image into CIELAB color space, as euclidean distances (used in the mean-shift procedure) in
    CIELAB color space correlate better with color changes perceived by the human eye. Retrieves features from converted
    image data. One of two types of features is used:
    (1) CIELAB color space (3D feature vector)
    (2) CIELAB color space + spatial information / coordinates (5D feature vector)
    Lastly, reshapes the segmented image to original image shape and converts it back to RGB color space.

    Parameters
    ----------
    img_rgb : image with shape [height]x[width]x[3]
    r : radius of shifting window in mean-shift procedure
    c : constant used for second speedup of mean-shift
    feature_type : FeatureType enumeration specifying the types of features to include in segmentation process

    Returns
    -------

    """
    # TODO smaller part of image for testing
    # img_rgb = img_rgb[-64:-32, 70:102, :]

    # preprocess image
    img_lab = rgb2lab(img_rgb)
    img_lab = utils.retrieve_features(img_lab, feature_type)
    # perform image segmentation using mean shift algorithm
    labels, peaks = mean_shift.ms_no_speedup(img_lab, r)
    # postprocess segmentation data
    segments = dict(zip(np.unique(labels), peaks))
    segmented = np.array([segments[l] if l in segments.keys() else l for l in labels])
    img_seg_lab = np.reshape(segmented[:, :3], img_rgb.shape)
    img_rgb_seg = lab2rgb(img_seg_lab)
    print("Found %s clusters." % len(segments))
    # plot clusters
    bgr_peaks = img_rgb_seg.reshape(img_rgb_seg.shape[0] * img_rgb_seg.shape[1], img_rgb_seg.shape[2])[..., ::-1]
    fig = plotclusters3D(img_lab.T, labels, bgr_peaks, rand_color=False)
    return img_rgb_seg, fig


def test_mean_shift():
    """
    Applies mean-shift algorithm to test data and plots found clusters/segments.

    Returns
    -------

    """
    data = utils.load_test_data()
    print("data shape: ", data.shape)

    print("Mean shift with no speedup...")
    labels, peaks = mean_shift.ms_no_speedup(data, r=2)
    print("mean shift - # cluster: %s, peaks: %s\n" % (np.unique(labels).size, peaks))
    plotclusters3D(data.T, labels, peaks.T)

    print("Mean shift with 1. speedup...")
    labels, peaks = mean_shift.ms_speedup1(data, r=2)
    print("1. speedup - # cluster: %s, peaks: %s\n" % (np.unique(labels).size, peaks))
    plotclusters3D(data.T, labels, peaks.T)

    print("Mean shift with 2. speedup...")
    labels, peaks = mean_shift.ms_speedup2(data, r=2, c=4)
    print("2. speedup - # cluster: %s, peaks: %s\n" % (np.unique(labels).size, peaks))
    plotclusters3D(data.T, labels, peaks.T)


if __name__ == '__main__':
    # test_mean_shift()
    # imgs = utils.load_images(filenames=["deer10.png", "img1.jpg", "img3.jpg"])
    # img_rgb = imgs[0]

    img = Image.img1
    img_rgb = utils.load_image(img)

    # Preprocessing (i.e. apply filter)
    img_rgb_gauss = utils.apply_filter(img_rgb, type='gaussian')
    img_rgb_median = utils.apply_filter(img_rgb, type='median')

    plt.imshow(img_rgb_gauss)
    plt.show()

    # Image segmentation
    r = 12
    c = 4
    feature_type = FeatureType.color  # color, color_spatial
    img_rgb_seg, cluster_fig = image_segmentation(img_rgb_gauss, r, c, feature_type)

    # show original and segmented image
    fig, ax = plt.subplots(3, 1, sharex=False, sharey=True)
    ax[0].imshow(img_rgb)
    ax[1].imshow(img_rgb_gauss)
    ax[2].imshow(img_rgb_seg)
    plt.show()

    cluster_fig.savefig(img.value['dest'] + "clusters3D.png")

    print("Mission accomplished.")
