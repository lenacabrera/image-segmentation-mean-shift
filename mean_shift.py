import numpy as np
from numba import njit
from tqdm import tqdm


def ms_no_speedup(data, r):

    n_data_points = data.shape[1]
    n_features = data.shape[0]

    labels = np.zeros((n_data_points,), dtype=np.int)
    peaks = np.zeros((n_data_points, n_features))

    unique_labels = 1

    # remaining data points
    for idx in tqdm(range(1, n_data_points)):
        peak, _ = find_peak(data, idx, r, threshold=0.01)
        peaks[idx] = peak

        if labels[idx] == 0:
            # check for similar peaks
            indices = np.argwhere(compute_distances(peak, peaks[:idx]) < r/2).flatten()
            if indices.size > 0:
                # assign same label for similar peaks
                labels[idx] = labels[indices[0]]
                for i in indices:
                    labels[i] = labels[indices[0]]
            else:
                # assign new label for new peaks
                labels[idx] = unique_labels
                unique_labels += 1

    # merge peaks associated to the same cluster
    peaks_merged = np.zeros((np.unique(labels).size, n_features))
    for i, label in enumerate(np.unique(labels)):
        idx = np.argwhere(labels == label)[0]
        peaks_merged[i] = peaks[idx]

    return labels, peaks_merged


def ms_speedup1(data, r):

    n_data_points = data.shape[1]
    n_features = data.shape[0]

    labels = np.zeros((n_data_points,), dtype=np.int)
    peaks = np.zeros((n_data_points, n_features))

    unique_labels = 1

    # remaining data points
    for idx in tqdm(range(1, n_data_points)):
        peak, window_indices = find_peak(data, idx, r, threshold=0.01)
        peaks[idx] = peak

        if labels[idx] == 0:
            # check for similar peaks
            indices = np.argwhere(compute_distances(peak, peaks[:idx]) < r/2).flatten()
            if indices.size > 0:
                # assign same label for similar peaks
                labels[idx] = labels[indices[0]]
                for i in indices:
                    labels[i] = labels[indices[0]]
            else:
                # assign new label for new peaks
                labels[idx] = unique_labels
                unique_labels += 1

        # assign points withing current point's window the same label
        labels[window_indices] = labels[idx]

    # merge peaks associated to the same cluster
    peaks_merged = np.zeros((np.unique(labels).size, n_features))
    for i, label in enumerate(np.unique(labels)):
        idx = np.argwhere(labels == label)[0]
        peaks_merged[i] = peaks[idx]

    return labels, peaks_merged


def ms_speedup2(data, r, c):

    n_data_points = data.shape[1]
    n_features = data.shape[0]

    labels = np.zeros((n_data_points,), dtype=np.int)
    peaks = np.zeros((n_data_points, n_features))

    unique_labels = 1

    # remaining data points
    for idx in tqdm(range(1, n_data_points)):
        peak, cpts = find_peak_opt(data, idx, r, threshold=0.01, c=c)
        peaks[idx] = peak

        if labels[idx] == 0:
            # check for similar peaks
            indices = np.argwhere(compute_distances(peak, peaks[:idx]) < r/2).flatten()
            if indices.size > 0:
                # assign same label for similar peaks
                labels[idx] = labels[indices[0]]
                for i in indices:
                    labels[i] = labels[indices[0]]
                # assign points within r/d distance of search path the same label
                path_indices = np.argwhere(cpts == 1)
                labels[path_indices] = labels[indices[0]]
            else:
                # assign new label for new peaks
                labels[idx] = unique_labels
                unique_labels += 1

    # merge peaks associated to the same cluster
    peaks_merged = np.zeros((np.unique(labels).size, n_features))
    for i, label in enumerate(np.unique(labels)):
        idx = np.argwhere(labels == label)[0]
        peaks_merged[i] = peaks[idx]

    return labels, peaks_merged


def find_peak(data, idx, r, threshold):
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
    center = data_point
    difference_peaks = threshold + 1

    # shift window to mean until convergence...
    while np.all(difference_peaks > threshold):
        # define window
        distances = compute_distances(center, data.T, 'euclidean')
        # retrieve points in window
        window_indices = np.argwhere(distances <= r).flatten()
        window = data.T[window_indices]
        # compute peak of window
        peak = np.mean(window, axis=0)
        # compare peak to previous peak
        # difference_peaks = np.abs(peak - center)
        difference_peaks = compute_distances(peak, center.reshape(1, -1))
        # shift window to mean
        center = peak

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
    center = data_point
    difference_peaks = threshold + 1
    cpts = np.zeros((data.shape[1],), dtype=np.int8)

    # shift window to mean until convergence...
    while np.all(difference_peaks > threshold):
        # define window
        distances = compute_distances(center, data.T, 'euclidean')
        # check if points are similar, i.e. distance between them is smaller r/c
        cpts[distances < r / c] = 1
        # retrieve points in window
        window_indices = np.argwhere(distances <= r).flatten()
        window = data.T[window_indices]
        # compute peak of window
        peak = np.mean(window, axis=0)
        # compare peak to previous peak
        # difference_peaks = np.abs(peak - center)
        difference_peaks = compute_distances(peak, center.reshape(1, -1))
        # shift window to mean
        center = peak

    return peak, cpts


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
    distances = np.array([euclidean_distance(data_point.reshape(1, -1), p.reshape(1, -1)) for p in data])
    return distances


@njit
def euclidean_distance(x, y) -> np.float32:
    return np.linalg.norm(x-y)