import scipy.io
import numpy as np
from scipy.spatial import distance
from numba import njit


@njit
def euclidean_distance(x, y) -> np.float32:
    return np.linalg.norm(x-y)


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
    # distances = np.array([scipy.spatial.distance.cdist(data_point.reshape(1, -1), p.reshape(1, -1), metric=metric) for p in data.T])
    # return distances.flatten()
    d = np.array([euclidean_distance(data_point.reshape(1, -1), p.reshape(1, -1)) for p in data.T])
    return d


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
    difference_peaks = np.ones((5,))

    # shift window to mean until convergence...
    while np.all(difference_peaks > threshold):
        # define window
        distances = compute_distances(data_point, data)
        # retrieve points in window
        window_indices = np.argwhere(distances <= r).flatten()
        window_indices = window_indices[window_indices != idx]
        neighbors = data.T[window_indices]
        # compute peak of window
        peak = np.mean(neighbors, axis=0)
        # compare peak to previous peak
        difference_peaks = np.abs(peak - data_point)
        # shift window to mean
        data_point = peak

    return peak, window_indices


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
    # retrieve data point
    data_point = data[:, idx]

    peak = data_point
    difference_peaks = np.ones((5,))
    cpts = np.zeros((data.shape[1], 1))

    # shift window to mean until convergence...
    while np.all(difference_peaks > threshold):
        # define window
        distances = compute_distances(data_point, data)
        # check if points are similar, i.e. distance between them is smaller r/c
        cpts[distances < r/c] = 1
        # retrieve points in window
        window_indices = np.argwhere(distances <= r).flatten()
        window_indices = window_indices[window_indices != idx]
        neighbors = data.T[window_indices]
        # compute peak of window
        peak = np.mean(neighbors, axis=0)
        # compare peak to previous peak
        difference_peaks = np.abs(peak - data_point)
        # shift window to mean
        data_point = peak

    return peak, cpts


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
    # call find_peak for each point and then assign a label to each point according to its peak
    # after each call compare peaks and merge similar ones (two peaks are the same if their distance is < r/2)
    # if found peak already exists in peaks, computed peak is discarded and is given label associated with peak in peaks

    peaks = np.zeros((data.shape[1], data.shape[0]))
    labels = np.zeros((data.shape[1], ))

    n_data_points = data.shape[1]
    unique_labels = 0

    for idx in range(n_data_points):
        print("idx: ", idx)
        peak, window_indices = find_peak(data, idx, r, threshold=0.01)

        # check if similar peak already found previously
        same_peaks = np.argwhere((np.abs(peaks[:idx+1] - np.vstack([peak] * (idx + 1))) < r / 2).all(axis=1))
        if same_peaks.size > 0:
            # assign same label (as previously found peak)
            label = labels[same_peaks[0, 0]]
            labels[window_indices] = labels[same_peaks[0]]
        else:
            # assign new label
            unique_labels += 1
            label = unique_labels

        labels[idx] = label
        peaks[idx] = peak

    return labels, peaks


def first_speedup(data, r, c):
# def mean_shift_opt(data, r, c):
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
    peaks = np.zeros((data.shape[1], data.shape[0]))
    labels = np.zeros((data.shape[1], ))

    n_data_points = data.shape[1]
    unique_labels = 0

    for idx in range(n_data_points):
        print("idx: ", idx)
        peak, window_indices = find_peak(data, idx, r, threshold=0.01)

        # check if similar peak already found previously
        same_peaks = np.argwhere((np.abs(peaks[:idx+1] - np.vstack([peak] * (idx + 1))) < r / 2).all(axis=1))
        if same_peaks.size > 0:
            # assign same label (as previously found peak)
            label = labels[same_peaks[0, 0]]
            labels[window_indices] = labels[same_peaks[0]]
        else:
            # assign new label
            unique_labels += 1
            label = unique_labels

        labels[idx] = label
        peaks[idx] = peak

    return labels, peaks


def second_speedup(data, r, c):
# def mean_shift_opt2(data, r, c):
    """
    Second speedup:
    Associates points that are within a distance of r/c of the search path with the converged peak

    Parameters
    ----------
    data : n-dimensional dataset
    r : search window radius
    c : constant value

    Returns
    -------

    """

    peaks = np.zeros((data.shape[1], data.shape[0]))
    labels = np.zeros((data.shape[1],))

    n_data_points = data.shape[1]
    unique_labels = 1

    for idx in range(n_data_points):
        print("idx: ", idx)
        peak, cpts = find_peak_opt(data, idx, r, threshold=0.01, c=c)
        # retrieve indices of points at distance r/c of search path
        indices = cpts[cpts == 1]

        # check if similar peak already found previously
        same_peaks = np.argwhere((np.abs(peaks[:idx + 1] - np.vstack([peak] * (idx + 1))) < r / 2).all(axis=1))
        if same_peaks.size > 0:
            # assign same label (as previously found peak)
            label = labels[same_peaks[0, 0]]
            labels[indices] = labels[same_peaks[0]]
        else:
            # assign new label
            label = unique_labels
            unique_labels += 1

        labels[idx] = label
        peaks[idx] = peak

    return labels, peaks