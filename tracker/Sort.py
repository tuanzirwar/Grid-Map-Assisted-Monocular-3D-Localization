import numpy as np
from scipy.spatial.distance import cdist
from .KalmanFilter import KalmanPointTracker
import lap


def linear_assignment(cost_matrix):
    _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
    return np.array([[y[i], i] for i in x if i >= 0])


def distance_mat(points1, points2):
    """
    Return distance matrix between points1 and points2
    """

    if not isinstance(points1, np.ndarray):
        points1 = np.array(points1).reshape(-1, 3)
    if not isinstance(points2, np.ndarray):
        points2 = np.array(points2).reshape(-1, 3)

    return cdist(points1, points2)


def associate_points_to_trackers(points, trackers, distance_threshold=0.3):
    """
    Assigns points to tracked object (both represented as bounding boxes)

    Returns 3 lists of matches, unmatched_points and unmatched_trackers
    """
    if (len(trackers) == 0):
        return np.empty((0, 2), dtype=int), np.arange(len(points)), np.empty((0, 4), dtype=int)

    distance_matrix = distance_mat(points, trackers)

    if min(distance_matrix.shape) > 0:
        a = (distance_matrix < distance_threshold).astype(np.int32)
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            matched_indices = linear_assignment(distance_matrix)
    else:
        matched_indices = np.empty(shape=(0, 2))

    unmatched_points = []
    for d, det in enumerate(points):
        if (d not in matched_indices[:, 0]):
            unmatched_points.append(d)
    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if (t not in matched_indices[:, 1]):
            unmatched_trackers.append(t)

    # filter out matched with  large distance
    matches = []
    for m in matched_indices:
        if (distance_matrix[m[0], m[1]] > distance_threshold):
            unmatched_points.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))
    if (len(matches) == 0):
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_points), np.array(unmatched_trackers)


class Sort(object):
    def __init__(self, max_age=1, min_hits=3, distance_threshold=0.3):
        """
        Sets key parameters for SORT
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.distance_threshold = distance_threshold
        self.trackers_map = {}
        self.frame_count = 0

    def update(self, observers=np.empty((0, 4))):
        """
        Params:
        points - a numpy array of points in the format [[x1,y1,z1,cls1],[x2,y2,z2,cls2],...]
        Requires: this method must be called once for each frame even with empty points (use np.empty((0, 5)) for frames without points).
        Returns the a similar array, where the last column is the object ID.

        NOTE: The number of objects returned may differ from the number of points provided.
        """
        self.frame_count += 1

        points_map = {}
        ret = np.zeros((observers.shape[0], 5))

        for i, point in enumerate(observers):
            cls = point[3]
            if cls not in points_map.keys():
                points_map[cls] = []
            points_map[cls].append(i)

        for cls, ids in points_map.items():

            # noval class trackers
            if cls not in self.trackers_map.keys():
                self.trackers_map[cls] = []

            # get predicted locations from existing trackers.
            trackers = self.trackers_map[cls]
            trks = np.zeros((len(trackers), 3))
            to_del = []
            for t, trk in enumerate(trks):
                pos = trackers[t].predict()[0]
                trk[:] = [pos[0], pos[1], pos[2]]
                if np.any(np.isnan(pos)):
                    to_del.append(t)
            trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
            for t in reversed(to_del):
                trackers.pop(t)

            points = observers[ids]
            matched_points, unmatched_points, unmatched_trks = associate_points_to_trackers(
                points[:, :3], trks, self.distance_threshold)

            # update matched_points trackers with assigned points
            for m in matched_points:
                trackers[m[1]].update(points[m[0], :3])
                trk = trackers[m[1]]
                ret[ids[m[0]]] = np.concatenate(
                    (trk.get_state()[0], [cls, trk.id+1]), axis=0).reshape(1, -1)

            # create and initialise new trackers for unmatched points
            for i in unmatched_points:
                trk = KalmanPointTracker(points[i, :])
                trackers.append(trk)
                ret[ids[i]] = np.concatenate(
                    (points[i, :3], [cls, trk.id+1]), axis=0).reshape(1, -1)

            # update unmatched trackers
            for i in unmatched_trks:
                trackers[i].predict()
                if (trk.time_since_update > self.max_age):
                    trackers.remove(trk)

            self.trackers_map[cls] = trackers

        # 不在观测范围内的tracker预测
        for cls in self.trackers_map.keys():
            if cls not in points_map.keys():
                for idx, _ in enumerate(self.trackers_map[cls]):
                    # remove dead tracklet
                    self.trackers_map[cls][idx].predict()
                    if (self.trackers_map[cls][idx].time_since_update > self.max_age):
                        self.trackers_map[cls].remove(_)

        if (len(ret) > 0):
            return ret
        return np.empty((0, 5))


def noise(points, noise_level=0.5):
    """
    Add noise to points
    """
    return points + np.random.randn(*points.shape) * noise_level


def plot(points, ax, title="", colors=[]):
    """
    Plot tracklets
    """
    xyzt = []
    for i in points:
        for j in i:
            xyzt.append([j[0], j[1], j[2], j[3]])

    xyzt = np.array(xyzt).reshape(-1, 4)
    ax.set_title(title)
    for i in xyzt:
        ax.scatter(i[0], i[1], color=colors[int(i[3])])
