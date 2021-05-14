import scipy.io
from scipy.spatial import distance
import numpy as np
from numba import njit
from tqdm import tqdm



def mean_shift_new(data, r):

    peaks = np.array([])
    labels = np.zeros((data.shape[1], ), dtype=np.int)

    n_data_points = data.shape[1]
    unique_labels = 1

    # first data point
    peak, _ = find_peak(data, 0, r, threshold=0.01)
    peaks = np.array([peak])
    labels[0] = unique_labels

    # remaining data points
    for idx in tqdm(range(1, n_data_points)):
        peak, _ = find_peak(data, idx, r, threshold=0.01)

        same_peaks_indices = np.argwhere(compute_distances(np.array([peak]), peaks) < (r/2)).flatten()
        if same_peaks_indices.size > 0:
            # similar peak already found previously, thus, merge peaks
            # peak_merged = np.mean(np.append(peaks[same_peaks_indices], np.array([peak]), axis=0), axis=0)
            peak_merged = peaks[same_peaks_indices[0]]
            peaks = np.delete(peaks, same_peaks_indices, axis=0)
            peaks = np.insert(peaks, same_peaks_indices[0], peak_merged, axis=0)
            # assign same label for similar peaks
            label = same_peaks_indices[0] + 1
            labels[idx] = label
            # same_peaks_indices = np.append(same_peaks_indices, idx)
            for i in same_peaks_indices:
                labels[labels == i+1] = label

        if labels[idx] == 0:
            # assign new label
            unique_labels = len(peaks) + 1
            labels[idx] = unique_labels
            # store new peak
            peaks = np.append(peaks, np.array([peak]), axis=0)

    return labels, peaks



def mean_shift_new1(data, r):

    peaks = np.array([])
    labels = np.zeros((data.shape[1], ), dtype=np.int)

    n_data_points = data.shape[1]
    unique_labels = 1

    # first data point
    peak, _ = find_peak(data, 0, r, threshold=0.01)
    peaks = np.array([peak])
    labels[0] = unique_labels

    # remaining data points
    for idx in tqdm(range(1, n_data_points)):
        peak, _ = find_peak(data, idx, r, threshold=0.01)

        same_peaks_indices = np.argwhere((np.abs(peaks - np.vstack([peak] * len(peaks))) < (r/2)).all(axis=1)).flatten()
        if same_peaks_indices.size > 0:
            # similar peak already found previously, thus, merge peaks
            peak_merged = np.mean(np.append(peaks[same_peaks_indices], np.array([peak]), axis=0), axis=0)
            peaks = np.delete(peaks, same_peaks_indices, axis=0)
            peaks = np.insert(peaks, same_peaks_indices[0], peak_merged, axis=0)
            # assign same label for similar peaks
            label = same_peaks_indices[0] + 1
            same_peaks_indices = np.append(same_peaks_indices, idx)
            labels[same_peaks_indices] = label

        if labels[idx] == 0:
            # assign new label
            unique_labels = len(peaks)
            labels[idx] = unique_labels
            # store new peak
            peaks = np.append(peaks, np.array([peak]), axis=0)

    return labels, peaks


# def mean_shift(data, r):
#     """
#     Computes peaks and corresponding labels of all points in the data.
#
#     Parameters
#     ----------
#     data : n-dimensional dataset
#     r : search window radius
#
#     Returns
#     -------
#
#     """
#     # call find_peak for each point and then assign a label to each point according to its peak
#     # after each call compare peaks and merge similar ones (two peaks are the same if their distance is < r/2)
#     # if found peak already exists in peaks, computed peak is discarded and is given label associated with peak in peaks
#
#     peaks = np.zeros((data.shape[1], data.shape[0]))
#     labels = np.zeros((data.shape[1], ))
#
#     n_data_points = data.shape[1]
#     unique_labels = 0
#
#     for idx in tqdm(range(n_data_points)):
#         peak, _ = find_peak(data, idx, r, threshold=0.01)
#
#         # check if similar peak already found previously
#         same_peaks = np.argwhere((np.abs(peaks[:idx+1] - np.vstack([peak] * (idx + 1))) < r / 2).all(axis=1))
#         if same_peaks.size > 0:
#             # assign same label (as previously found peak)
#             label = labels[same_peaks[0, 0]]
#             # labels[window_indices] = labels[same_peaks[0]] # TODO merge
#         else:
#             # assign new label
#             unique_labels += 1
#             label = unique_labels
#
#         labels[idx] = label
#         peaks[idx] = peak
#
#     return labels, peaks


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

    peaks = []
    labels = np.zeros((data.shape[1], ), dtype=np.int)

    n_data_points = data.shape[1]
    unique_labels = 1

    # first data point
    peak, _ = find_peak(data, 0, r, threshold=0.01)
    peaks.append(peak)
    labels[0] = unique_labels

    # remaining data points
    for idx in tqdm(range(1, n_data_points)):
        peak, _ = find_peak(data, idx, r, threshold=0.01)

        # check if similar peak already found previously
        same_peaks = np.argwhere((np.abs(np.array(peaks) - np.vstack([peak] * len(peaks))) < r / 2).all(axis=1))
        if same_peaks.size > 0:
            # assign same label (merge with previously found peak)
            labels[idx] = same_peaks[0, 0] + 1
            # labels[idx] = labels[same_peaks[0, 0]]
            labels[same_peaks[0]] = same_peaks[0, 0] + 1
            if 1 < same_peaks.size <= len(peaks):
                peaks = list(np.delete(np.array(peaks), list(same_peaks[1:, 0]), axis=0))
        if labels[idx] == 0:
        # else:
            # assign new label
            unique_labels = len(peaks)
            labels[idx] = unique_labels
            # store new peak
            peaks.append(peak)

    return labels, np.array(peaks)
    # return labels, peaks


# def mean_shift(data, r):
#     """
#     Computes peaks and corresponding labels of all points in the data.
#
#     Parameters
#     ----------
#     data : n-dimensional dataset
#     r : search window radius
#
#     Returns
#     -------
#
#     """
#     # call find_peak for each point and then assign a label to each point according to its peak
#     # after each call compare peaks and merge similar ones (two peaks are the same if their distance is < r/2)
#     # if found peak already exists in peaks, computed peak is discarded and is given label associated with peak in peaks
#
#     ppeaks = np.zeros((data.shape[1], data.shape[0]))
#     peaks = []
#     labels = np.zeros((data.shape[1], ), dtype=np.int)
#
#     n_data_points = data.shape[1]
#     unique_labels = 1
#
#     # first data point
#     peak, _ = find_peak(data, 0, r, threshold=0.01)
#     peaks.append(peak)
#     ppeaks[0, :] = peak
#     labels[0] = unique_labels
#
#     # remaining data points
#     for idx in tqdm(range(1, n_data_points)):
#         peak, _ = find_peak(data, idx, r, threshold=0.01)
#
#         if labels[idx] == 0:
#             # check if similar peak already found previously
#             # same_peaks = np.argwhere((np.abs(np.array(peaks) - np.vstack([peak] * unique_labels)) < r / 2).all(axis=1))
#             same_peaks = np.argwhere((np.abs(ppeaks[:idx] - np.vstack([peak] * idx)) < r / 2).all(axis=1))
#             if same_peaks.size > 0:
#                 # assign same label (merge with previously found peak)
#                 # labels[idx] = same_peaks[0, 0] + 1
#                 labels[idx] = labels[same_peaks[0, 0]]
#                 ppeaks[idx, :] = ppeaks[same_peaks[0, 0]]
#                 labels[same_peaks[0]] = labels[same_peaks[0, 0]]
#
#             if labels[idx] == 0:
#                 # assign new label
#                 unique_labels += 1
#                 labels[idx] = unique_labels
#                 ppeaks[idx, :] = peak
#                 # store new peak
#                 peaks.append(peak)
#
#     pppeaks = np.unique(ppeaks, axis=0)
#
#     # return labels, np.array(peaks)
#     return labels, pppeaks


def mean_shift_speedup1(data, r):
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
    peaks = []
    labels = np.zeros((data.shape[1],))

    n_data_points = data.shape[1]
    unique_labels = 1

    # first data point
    peak, _ = find_peak(data, 0, r, threshold=0.01)
    peaks.append(peak)
    labels[0] = unique_labels

    # remaining data points
    for idx in tqdm(range(1, n_data_points)):
        peak, window_indices = find_peak(data, idx, r, threshold=0.01)

        # check if similar peak already found previously
        same_peaks = np.argwhere((np.abs(np.array(peaks) - np.vstack([peak] * unique_labels)) < r / 2).all(axis=1))
        if same_peaks.size > 0:
            # assign same label (merge with previously found peak)
            labels[idx] = same_peaks[0, 0] + 1
            labels[window_indices] = same_peaks[0, 0] + 1
        if labels[idx] == 0:
            # assign new label
            unique_labels += 1
            labels[idx] = unique_labels
            # store new peak
            peaks.append(peak)

    return labels, np.array(peaks)


def mean_shift_speedup2(data, r, c):
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

    peaks = []
    ppeaks = np.zeros((data.shape[1], data.shape[0]))
    labels = np.zeros((data.shape[1],))

    n_data_points = data.shape[1]
    unique_labels = 1

    # first data point
    peak, _ = find_peak(data, 0, r, threshold=0.01)
    peaks.append(peak)
    labels[0] = unique_labels

    # remaining data points
    for idx in tqdm(range(1, n_data_points)):
        if labels[idx] == 0:
            peak, cpts = find_peak_opt(data, idx, r, threshold=0.01, c=c)
            # retrieve indices of points at distance r/c of search path
            indices = np.argwhere(cpts == 1)

            # check if similar peak already found previously
            # same_peaks = np.argwhere((np.abs(np.array(peaks) - np.vstack([peak] * unique_labels)) < r / 2).all(axis=1))
            same_peaks = np.argwhere((np.abs(ppeaks[:idx] - np.vstack([peak] * idx)) < r / 2).all(axis=1))
            if same_peaks.size > 0:
                # assign same label (merge with previously found peak)
                labels[idx] = same_peaks[0, 0] + 1
                labels[indices] = same_peaks[0, 0] + 1
            if labels[idx] == 0:
                # assign new label
                unique_labels += 1
                labels[indices] = unique_labels
                # store new peak
                peaks.append(peak)
                ppeaks[indices] = peak

    pppeaks = np.unique(ppeaks)

    # return labels, np.array(peaks)
    return labels, pppeaks


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
    difference_peaks = np.ones((5,))

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
        difference_peaks = np.abs(peak - center)
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

    peak = data_point
    difference_peaks = np.ones((5,))
    cpts = np.zeros((data.shape[1], ), dtype=np.int8)

    # shift window to mean until convergence...
    while np.all(difference_peaks > threshold):
        # define window
        distances = compute_distances(data_point, data)
        # check if points are similar, i.e. distance between them is smaller r/c
        cpts[distances < r/c] = 1
        # retrieve points in window
        window_indices = np.argwhere(distances <= r).flatten()
        neighbors = data.T[window_indices]
        # compute peak of window
        peak = np.mean(neighbors, axis=0)
        # compare peak to previous peak
        difference_peaks = np.abs(peak - data_point)
        # shift window to mean
        data_point = peak

    return peak, cpts


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
    distances = np.array([euclidean_distance(data_point.reshape(1, -1), p.reshape(1, -1)) for p in data])
    return distances