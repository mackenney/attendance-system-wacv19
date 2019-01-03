import numpy as np
import os


def invert_db(db):
    """
    Takes a database (array of dictionaries) and turns it in a dictionary of
    arrays.

    :param db: an array of dictionaries
    :returns: a dictionary of arrays
    """
    l = []
    for d in db:
        l.append(d)
    keys = []
    for k in l[0].keys():
        keys.append(k)
    data = []
    for i in range(len(keys)):
        data += [[]]
    for d in db:
        for i, k in enumerate(keys):
            data[i] += [d[k]]
    for i in range(len(data)):
        data[i] = np.array(data[i])
    dicc = {}
    for i in range(len(data)):
        dicc.update({keys[i]: data[i][0]})
    return dicc


def get_attendance_matrix(ids, day, num_ids=67, num_days=25):
    """
    Returns the attendance matrix.
    each entry in them represents a face. 
    ids marks its identity and day the day that the face was recorded.

    :param ids: array of integers
    :param day: array of integers, from the same length of ids
    :param num_ids: integer, number of identities
    :param num_days: integer, number of days
    :returns: a num_ids by num_days boolean matrix
    """
    matrix = np.zeros((num_ids, num_days), dtype=bool)
    for id_ in range(num_ids):
        for d in range(num_days):
            if np.count_nonzero(ids[np.nonzero(day == d)] == id_) > 0:
                matrix[id_, d] = True
    return matrix


def cross_matrix(X, Z):
    """
    Normalizes and multiply the matrices.
    :param X: numpy ndarray. enrollment matrix of features
    :param Z: numpy ndarray. Test matrix of features.
    :returns: a numpy ndarray.
    """
    X = np.asarray(X, dtype=np.float64)
    Z = np.asarray(Z, dtype=np.float64)
    for i in range(len(X)):
        n = np.linalg.norm(X[i])
        if n > 0:
            X[i] = X[i] / n
    for i in range(len(Z)):
        n = np.linalg.norm(Z[i])
        if n > 0:
            Z[i] = Z[i] / n
    out = X @ Z.T
    return out


def predict_attendace(X, y, Z, theta):
    """ 3.5 in paper.
    Method for predicting attendace. every max match between a enroll
    identity and a face is selected as a positive attendance if its larger than
    a theshold theta.
    :param X: ndarray, enrolled feature vectors.
    :param y: ndarray, ids of enrolled
    :param Z: ndarray, feat. vectors of incognitos
    :returns: ndarray, contains every id that matched.
    """
    cross = cross_matrix(X, Z)
    best = np.argmax(cross, axis=0)
    vals = np.zeros(len(Z))
    for i in range(len(vals)):
        vals[i] = cross[best[i], i]
    cualifies = vals > theta
    ids = np.unique(y[best[cualifies]])
    return ids


def simple_experiment(feats, y, day, enroll_feats, y_enroll, num_post_enrollment=3,
                      theta=1):
    """ Algorithm used to build Table 3.
    :param feats: query feats
    :param y: query identity/ ground thuth
    :param day: day marker array
    :param enroll_feats: enroll feats
    :param y_enroll: enroll identity
    :param num_post_enrollment: number of days that are progressively added to the
    :param theta: threshold.
    """
    predicted_by_day = []
    for d in range(max(day) + 1):
        post_enrollment = []
        y_post_enroll = []
        post_enrollment.append(enroll_feats)
        y_post_enroll.append(y_enroll)
        for i in range(1, num_post_enrollment + 1):
            if i <= d:
                post_enrollment.append(feats[day == i-1])
                y_post_enroll.append(y[day == i-1])
        enrolled = np.concatenate(post_enrollment)
        y_enrolled = np.concatenate(y_post_enroll)
        predicted_by_day.append(predict_attendace(enrolled, y_enrolled, feats[day == d],
                                                  theta))
    M = np.zeros((max(y_enroll) + 1, max(day) + 1), dtype=bool)
    for d in range(len(M[0])):
        for s in predicted_by_day[d]:
            M[s, d] = 1
    return M


def enroll_experiment(X, y, d, W, z, theta=.5):
    """ Experimental Protocol
    :param X: ndarray, query feature vectors
    :param y: ndarray, ground truth
    :param d: ndarray, corresponding day for each feature vector in X
    :param W: ndarray, enrolled feature vectors
    :param z: ndarray, identity of enrolled vectors
    :param theta: float, threshold
    :returns: tuple of ndarrays, mean errors and attendance matrices.
    """
    true_M = get_attendance_matrix(y, d)
    iteration = []
    matrix = []
    num_days = max(d) + 1
    for num_enroll in range(num_days):
        predicted_by_day = []
        post_enrollment = []
        y_post_enroll = []
        post_enrollment.append(W)
        y_post_enroll.append(z)
        for i in range(num_enroll):
            post_enrollment.append(X[d == i])
            y_post_enroll.append(y[d == i])
        enrolled = np.concatenate(post_enrollment)
        y_enrolled = np.concatenate(y_post_enroll)
        for k in range(num_enroll, num_days):
            predicted_by_day.append(predict_attendace(enrolled, y_enrolled, X[d == k],
                                                      theta))
        M_ = true_M[:, num_enroll:]
        M = np.zeros_like(M_, dtype=bool)
        for day in range(num_days - num_enroll):
            for id_ in predicted_by_day[day]:
                M[id_, day] = True
        error_M = M_ != M
        mean_error = np.count_nonzero(error_M) / len(np.ravel(error_M))
        iteration.append(mean_error)
        matrix.append(M)
    return iteration, matrix


def report_results(X, y, d, W, z, n, theta):
    """
    helper function that reports the error for the simple_experiment
    :param X: query feats
    :param y: query identity/ ground thuth
    :param d: day marker array
    :param W: enroll feats
    :param z: enroll identity
    :param n: number of days that are progressively added to the
    :param theta: threshold.
    :returns: float, error.
    """
    M = simple_experiment(X, y, d, W, z, n, theta)
    true_M = get_attendance_matrix(y, d)
    error = np.array(len(M.T[0]))
    error_M = M != true_M
    error = np.count_nonzero(error_M, axis=0) / len(M.T[0])
    return error


if __name__ == '__main__':
    ### Here an example of the usage of the code for one set of features.
    # load the database
    db = np.load('DB/data/query_database.npy')
    db2 = invert_db(db)
    # set the relevant variables.
    y = db2['identity'] # ground truth
    d = db2['day'] # markers for days
    # Load the features
    X = np.load('DB/query_feats/q_dlib.npy')
    W = np.load('DB/enroll_feats/e_dlib.npy')
    z = np.arange(len(W)) # identities for the enroll photos
    errors, matrices = enroll_experiment(X, y, d, W, z, theta=0.5)


