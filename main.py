from enum import Enum
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2lab, lab2rgb
import utils
import mean_shift
from plotclusters3D import plotclusters3D


class Image(Enum):
    """
    Specifies the input image.
    """
    img1 = {'src': "img/img1.jpg", 'dest3': "results/3D/img1/", 'dest5': "results/5D/img1/"}
    img2 = {'src': "img/img2.jpg", 'dest3': "results/3D/img2/", 'dest5': "results/5D/img2/"}
    img3 = {'src': "img/img3.jpg", 'dest3': "results/3D/img3/", 'dest5': "results/5D/img3/"}
    img4 = {'src': "img/deer10.png", 'dest3': "results/3D/img4/", 'dest5': "results/5D/img4/"}


class FeatureType(Enum):
    """
    Specifies the type of image features - either 3D color features or 5D color and spatial feature.
    """
    color = 3             # uses only color space
    color_spatial = 5     # uses color and spatial information


class Filter(Enum):
    """
    Specifies the filter to be applied to image.
    """
    none = 0
    gauss = 1
    median = 2


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
    labels, peaks = mean_shift.ms_speedup2(img_lab, r, c)
    # postprocess segmentation data
    segments = dict(zip(np.unique(labels), peaks))
    segmented = np.array([segments[l] if l in segments.keys() else l for l in labels])
    img_seg_lab = np.reshape(segmented[:, :3], img_rgb.shape)
    img_rgb_seg = lab2rgb(img_seg_lab)
    print("Found %s clusters." % len(segments))
    # plot clusters
    # bgr_peaks = img_rgb_seg.reshape(img_rgb_seg.shape[0] * img_rgb_seg.shape[1], img_rgb_seg.shape[2])[..., ::-1]
    # fig = plotclusters3D(img_lab.T, labels, bgr_peaks, rand_color=False)
    # return img_rgb_seg, fig, len(segments)
    return img_rgb_seg, 0, len(segments)


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
    test_mean_shift()

    # Configuration
    img = Image.img4
    feature_type = FeatureType.color  # color, color_spatial
    fltr = Filter.none
    r = 2
    c = 4

    img_rgb = utils.load_image(img)

    img_rgb_f = img_rgb
    if fltr.name == Filter.gauss.name:
        img_rgb_f = utils.apply_filter(img_rgb, type='gaussian')
    if fltr.name == Filter.gauss.name:
        img_rgb_f = utils.apply_filter(img_rgb, type='median')

    # plt.imshow(img_rgb_f)
    # plt.show()

    # Image segmentation
    img_rgb_seg, cluster_fig, n_peaks = image_segmentation(img_rgb, r, c, feature_type)

    if feature_type.value == 3:
        res_dir = img.value['dest3']
    if feature_type.value == 5:
        res_dir = img.value['dest5']

    # cluster_fig.savefig(res_dir + "cluster_r%s_c%s_p%s" % (r, c, n_peaks) + ".png")

    # show original and segmented image
    # fig, ax = plt.subplots(3, 1, sharex=False, sharey=True)
    # ax[0].imshow(img_rgb)
    # ax[1].imshow(img_rgb_f)
    # ax[2].imshow(img_rgb_seg)
    # plt.show()

    plt.imshow(img_rgb_seg)
    plt.title("r = %s   c = %s   p = %s" % (r, c, n_peaks))
    plt.savefig(res_dir + "seg_r%s_c%s_p%s" % (r, c, n_peaks) + ".png")
    plt.show()

    print("Mission accomplished.")
