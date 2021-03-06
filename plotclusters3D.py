import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def plotclusters3D(data, labels, peaks, rand_color=True):
    """
    Plots the modes of the given image data in 3D by coloring each pixel according to its corresponding peak.

    Args:
        data: image data in the format [number of pixels]x[feature vector].
        labels: a list of labels, one for each pixel.
        peaks: a list of vectors, whose first three components can be interpreted as BGR values.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    bgr_peaks = np.array(peaks[:, 0:3], dtype=float)
    rgb_peaks = bgr_peaks[..., ::-1]
    rgb_peaks /= 255.0
    rgb_peaks = np.array(peaks[:, 0:3], dtype=float)
    for idx, peak in enumerate(tqdm(rgb_peaks)):
        if rand_color:
            color = np.random.uniform(0, 1, 3)
        else:
            color = peak
        cluster = data[np.where(labels == idx)[0]].T
        ax.scatter(cluster[0], cluster[1], cluster[2], c=[color], s=.5)
    fig.show()
    return fig
