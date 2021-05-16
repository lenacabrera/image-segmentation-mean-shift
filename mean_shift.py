import numpy as np
from numba import njit
from tqdm import tqdm


def ms_no_speedup(data, r):
    """
    Performs the mean-shift algorithm to cluster pixel of in an image.
    Calls find_peak for each point and then assigns a label to each point according to its associated peak. Compares
    peaks after each call of find_peak and merges similar peaks. Two peaks are considered to be the similar if the
    distance between them is smaller than r/2. If the peak of a data point already exists then its computed peak is
    discarded and it is given the label of the associated peak in peaks.

    Parameters
    ----------
    data : pixel matrix with shape [#features]x[#pixels]
    r : radius of shifting spherical window

    Returns
    -------
    labels : cluster vector with a label for each pixel with shape [#pixels]
    peaks : peaks in density dist. / cluster centers found during mean-shift procedure, shape [#clusters]x[#features]

    """
    n_data_points = data.shape[1]
    n_features = data.shape[0]

    peaks = np.zeros((1, n_features))
    labels = np.zeros((n_data_points,), dtype=np.int)

    peak_indices = np.zeros((1,), dtype=np.int)
    unique_labels = 1

    # first data point
    peak, _ = find_peak(data, 0, r, threshold=0.01)
    peaks[0, :] = peak
    labels[0] = unique_labels
    unique_labels += 1

    # remaining data points
    for idx in tqdm(range(1, n_data_points)):
        peak, _ = find_peak(data, idx, r, threshold=0.01)

        # check for similar peaks
        indices = np.argwhere(compute_distances(peak, peaks) < r/2).flatten()
        if indices.size > 0:
            # assign same label for similar peaks
            labels[idx] = labels[peak_indices[indices[0]]]
            # labels[indices] = labels[idx]
        if labels[idx] == 0:
            # assign new label for new peaks
            labels[idx] = unique_labels
            unique_labels += 1
            # store new peak
            peaks = np.append(peaks, np.array([peak]), axis=0)
            # store index of new labeled peak
            peak_indices = np.append(peak_indices, idx)

    return labels, peaks


def ms_speedup1(data, r):
    """
    First speedup of mean-shift procedure: "Basin of attraction"
    Upon finding a peak, associates each data point that is at a distance <= r from the peak with the cluster defined by
    that peak. Based on the intuition that points that are within one window size distance from the peak will with high
    probability converge to that peak.

    Otherwise same as ms_no_speedup:
    Performs the mean-shift algorithm to cluster pixel of in an image.
    Calls find_peak for each point and then assigns a label to each point according to its associated peak. Compares
    peaks after each call of find_peak and merges similar peaks. Two peaks are considered to be the similar if the
    distance between them is smaller than r/2. If the peak of a data point already exists then its computed peak is
    discarded and it is given the label of the associated peak in peaks.

    Parameters
    ----------
    data : pixel matrix with shape [#features]x[#pixels]
    r : radius of shifting spherical window

    Returns
    -------
    labels : cluster vector with a label for each pixel with shape [#pixels]
    peaks : peaks in density dist. / cluster centers found during mean-shift procedure, shape [#clusters]x[#features]

    """
    n_data_points = data.shape[1]
    n_features = data.shape[0]

    peaks = np.zeros((1, n_features))
    labels = np.zeros((n_data_points,), dtype=np.int)

    peak_indices = np.zeros((1,), dtype=np.int)
    unique_labels = 1

    # first data point
    peak, window_indices = find_peak(data, 0, r, threshold=0.01)
    peaks[0, :] = peak
    labels[0] = unique_labels
    labels[window_indices] = labels[0]
    unique_labels += 1

    # remaining data points
    for idx in tqdm(range(1, n_data_points)):

        if labels[idx] == 0:
            peak, window_indices = find_peak(data, idx, r, threshold=0.01)

            # check for similar peaks
            indices = np.argwhere(compute_distances(peak, peaks) < r/2).flatten()
            if indices.size > 0:
                # assign same label for similar peaks
                labels[idx] = labels[peak_indices[indices[0]]]
                labels[indices] = labels[idx]
                # assign points within current point's window the same label
                labels[window_indices] = labels[idx]
            else:
                # assign new label for new peaks
                labels[idx] = unique_labels
                # assign points within current point's window the same label
                labels[window_indices] = unique_labels
                unique_labels += 1
                # store new peak
                peaks = np.append(peaks, np.array([peak]), axis=0)
                # store index of new labeled peak
                peak_indices = np.append(peak_indices, idx)

    peaks = peaks[np.unique(labels) - 1]
    clusters = dict(zip(np.unique(labels), range(np.unique(labels).size)))
    labels = np.array([clusters[l] if l in clusters.keys() else l for l in labels])

    return labels, peaks


def ms_speedup2(data, r, c):
    """
    Second speedup of mean-shift procedure:
    Points within a distance r/c of the search path are associated with the converged peak, where c is some constant
    value. These points are retrieved during a call of find_peak_opt.

    Otherwise same as ms_no_speedup:
    Performs the mean-shift algorithm to cluster pixel of in an image.
    Calls find_peak for each point and then assigns a label to each point according to its associated peak. Compares
    peaks after each call of find_peak and merges similar peaks. Two peaks are considered to be the similar if the
    distance between them is smaller than r/2. If the peak of a data point already exists then its computed peak is
    discarded and it is given the label of the associated peak in peaks.

    Parameters
    ----------
    data : pixel matrix with shape [#features]x[#pixels]
    r : radius of shifting spherical window
    c : constant

    Returns
    -------
    labels : cluster vector with a label for each pixel with shape [#pixels]
    peaks : peaks in density dist. / cluster centers found during mean-shift procedure, shape [#clusters]x[#features]

    """
    n_data_points = data.shape[1]
    n_features = data.shape[0]

    peaks = np.zeros((1, n_features))
    labels = np.zeros((n_data_points,), dtype=np.int)

    peak_indices = np.zeros((1,), dtype=np.int)
    unique_labels = 1

    # first data point
    peak, window_indices, cpts = find_peak_opt(data, 0, r, threshold=0.01, c=c)
    peaks[0, :] = peak
    labels[0] = unique_labels
    path_indices = np.argwhere(cpts == 1)
    labels[window_indices] = unique_labels
    labels[path_indices] = unique_labels
    unique_labels += 1

    # remaining data points
    for idx in tqdm(range(1, n_data_points)):

        if labels[idx] == 0:
            peak, window_indices, cpts = find_peak_opt(data, idx, r, threshold=0.01, c=c)
            path_indices = np.argwhere(cpts == 1)

            # check for similar peaks
            indices = np.argwhere(compute_distances(peak, peaks) < (r / 2)).flatten()
            if indices.size > 0:
                # assign same label for similar peaks
                labels[idx] = labels[peak_indices[indices[0]]]
                labels[indices] = labels[idx]
                # assign points within current point's window the same label
                labels[window_indices] = labels[idx]
                # assign points within r/d distance of search path the same label
                labels[path_indices] = labels[idx]
            else:
                # assign new label for new peaks
                labels[idx] = unique_labels
                # assign points within current point's window the same label
                labels[window_indices] = unique_labels
                # assign points within r/d distance of search path the same label
                labels[path_indices] = unique_labels
                unique_labels += 1
                # store new peak
                peaks = np.append(peaks, np.array([peak]), axis=0)
                # store index of new labeled peak
                peak_indices = np.append(peak_indices, idx)

    peaks = peaks[np.unique(labels) - 1]
    clusters = dict(zip(np.unique(labels), range(np.unique(labels).size)))
    labels = np.array([clusters[l] if l in clusters.keys() else l for l in labels])

    return labels, peaks


def find_peak(data, idx, r, threshold):
    """
    Computes density peak associated with a point whose column index is provided.

    Parameters
    ----------
    data : n-dimensional dataset
    idx : column index of data point
    r : search window radius

    Returns
    -------
    peak : vector of converged peak with shape [#features]
    window_indices : indices vector of points within converged window

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
        difference_peaks = compute_distances(peak, center.reshape(1, -1))
        # shift window to mean
        center = peak

    return peak, window_indices


def find_peak_opt(data, idx, r, threshold, c):
    """
    Computes density peak associated with a point whose column index is provided with speedup:
    associates points that are within a distance of r/c of the search path with the converged peak.

    Parameters
    ----------
    data : n-dimensional dataset
    idx : column index of data point
    r : search window radius
    threshold : threshold
    c : constant value

    Returns
    -------
    peak : vector of converged peak with shape [#features]
    window_indices : indices vector of points within converged window
    cpts : vector storing 1 for points within a distance of r/c of search path with the converged peak, shape [#pixels]

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
        cpts[distances < (r / c)] = 1
        # retrieve points in window
        window_indices = np.argwhere(distances <= r).flatten()
        window = data.T[window_indices]
        # compute peak of window
        peak = np.mean(window, axis=0)
        # compare peak to previous peak
        difference_peaks = compute_distances(peak, center.reshape(1, -1))
        # shift window to mean
        center = peak

    return peak, window_indices, cpts


def compute_distances(data_point, data, metric='euclidean'):
    """
    Computes the euclidean distances between a data point and a set of data points.

    Parameters
    ----------
    data_point : pixel
    data : n pixels
    metric : distance metric

    Returns
    -------
    distances : vector of distances with shape [n pixels]

    """
    distances = np.array([euclidean_distance(data_point.reshape(1, -1), p.reshape(1, -1)) for p in data])
    return distances


@njit
def euclidean_distance(x, y) -> np.float32:
    """
    Computes the euclidean distance between two vectors, here pixels.

    Parameters
    ----------
    x : pixel vector
    y : pixel vector

    Returns
    -------
    float : euclidean distance between x an y

    """
    return np.linalg.norm(x-y)