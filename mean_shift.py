import numpy as np


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
    # call find_peak for each point and then assign a label to each point according to its peak
    # after each call compare peaks and merge similar ones (two peaks are the same if their distance is < r/2)
    # if found peak already exists in peaks, computed peak is discarded and is given label associated with peak in peaks

    peaks = np.zeros((data.shape[1], data.shape[0]))
    labels = np.zeros((data.shape[1], ))

    n_data_points = data.shape[1]
    unique_labels = 0

    for idx in range(n_data_points):
        peak, merge_peaks = find_peak(data, idx, r, threshold=3)#0.01)

        # check if similar peak already found previously
        same_peaks = np.argwhere((np.abs(peaks - np.vstack([peak] * n_data_points)) < r / 2).all(axis=0))
        if same_peaks.size > 0:
            # assign same label (as previously found peak)
            label = labels[same_peaks[0]]
        else:
            # assign new label
            label = unique_labels
            unique_labels += 1

        labels[idx] = label
        peaks[idx] = peak

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

