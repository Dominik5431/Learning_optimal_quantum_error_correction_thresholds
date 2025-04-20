import numpy as np
from scipy.optimize import minimize, brentq
from scipy.interpolate import UnivariateSpline

"""
This file contains several functions used for data analysis.
"""


def gaussian(x, A, mu, sigma):
    return A * np.exp(-(x - mu) ** 2 / sigma ** 2)


def sigmoid_der(x, A, mu, sigma):
    # Sigmoid derivative
    return sigmoid(x, A, mu, sigma) * (1 - sigmoid(x, A, mu, sigma))


def sigmoid(x, A, mu, sigma):
    return 1 / (1 + np.exp(-sigma * (x - mu)))


def linear(x, m, n):
    return m * x + n


def smooth(arr, m):
    """
    Smoothes an array
    :param arr: Array to smoooth.
    :param m: Smoothing parameter.
    :return: Smoothed array.
    """
    res = arr.copy()
    for i in np.arange(int((m - 1) / 2), len(arr) - int((m - 1) / 2)):
        res[i] = 1 / m * np.sum(arr[i - int((m - 1) / 2): i + int((m - 1) / 2) + 1])
    return res


def data_collapse(noises, dictionary):
    """
    Performs data collapse.
    :param noises: noise values
    :param dictionary: dictionary: keys: distances, labels: probabilties for ordered/unordered as 2-tuple
    :return:
    """
    distances_keys = dictionary.keys()
    distances = np.array(list(distances_keys)).astype(int)
    predictions = np.zeros((len(distances), len(noises), 2))
    errors = np.zeros((len(distances), len(noises), 2))
    for i, key in enumerate(distances_keys):
        predictions[i], errors[i] = dictionary[key]

    def w(x, y, d, xp, yp, dp, xpp, ypp, dpp):
        ybar = ((xpp - x) * yp - (xp - x) * ypp) / (xpp - xp)
        deltasquared = d ** 2 + ((xpp - x) * dp / (xpp - xp)) ** 2 + ((xp - x) * dpp / (xpp - xp)) ** 2
        return (y - ybar) ** 2 / deltasquared

    def scale_x(p, pc, nu, d):
        return (p - pc) * d ** (1 / nu)

    # does not work too well - does not give the expected minimum of close to 1 of the objective that
    # is stated in the paper
    def objective_zabalo(x):
        pc = x[0]
        nu = x[1]
        result = 0
        for k in np.arange(len(noises)):  # noises
            # print(noises[k])
            for h in np.arange(1, len(distances) - 1):  # different distances
                # print(distances[l])
                result += w(scale_x(noises[k], pc, nu, distances[h]), predictions[h, k, 0], errors[h, k, 0],
                            scale_x(noises[k], pc, nu, distances[h - 1]), predictions[h - 1, k, 0],
                            errors[h - 1, k, 0],
                            scale_x(noises[k], pc, nu, distances[h + 1]), predictions[h + 1, k, 0],
                            errors[h + 1, k, 0])
                result += w(scale_x(noises[k], pc, nu, distances[h]), predictions[h, k, 1], errors[h, k, 1],
                            scale_x(noises[k], pc, nu, distances[h - 1]), predictions[h - 1, k, 1],
                            errors[h - 1, k, 1],
                            scale_x(noises[k], pc, nu, distances[h + 1]), predictions[h + 1, k, 1],
                            errors[h + 1, k, 1])
        result *= 1 / (2 * (len(distances) - 2) * len(noises))
        return result

    def objective(x):
        pc = x[0]
        c = x[1]
        d = x[2]
        q = 1
        result = 0
        for k in np.arange(len(distances)):
            for l in np.arange(k, len(distances)):
                for h in np.arange(len(noises)):
                    result += dist ** (-d) * predictions

    x0 = np.array([0.1089, 1.])
    res = minimize(objective_zabalo, x0, method='nelder-mead', options={'xatol': 1e-8, 'disp': True})
    print(res.x)
    print(objective_zabalo(res.x))
    return res


def find_crossing(noises, predictions):
    """
    Find the crossing by doing spline interpolation of two given arrays.
    :param noises: Noise strengths.
    :param predictions: shape (len(noises)x2): predictions for class 0 and 1
    :return:
    """

    def temp(x1, x2, y10, y20, y11, y21):
        m0 = (y20 - y10) / (x2 - x1)
        m1 = (y21 - y11) / (x2 - x1)
        n0 = -m0 * x1 + y10
        n1 = -m1 * x1 + y11
        crossing = (n1 - n0) / (m0 - m1)
        if x1 < crossing < x2:
            print(crossing, " x1 ", x1, " x2 ", x2)
            return crossing
        else:
            return -1

    crossings = []
    for i in np.arange(len(noises) - 1):
        cross = temp(noises[i], noises[i + 1], predictions[i, 0], predictions[i + 1, 0], predictions[i, 1],
                     predictions[i + 1, 1])
        if cross != -1:
            crossings.append(cross)
    print(crossings)
    pcritical = np.mean(np.array(crossings))
    return pcritical


def get_pcs(dictionary, noises):
    """

    :param dictionary:
    :param noises:
    :return: result: 1st row: 1/distance, 2nd row: respective pcritical from crossing
    """
    distances_keys = dictionary.keys()
    distances = np.array(list(distances_keys)).astype(int)
    predictions = np.zeros((len(distances), len(noises), 2))
    predictions_err = np.zeros((len(distances), len(noises), 2))
    for i, key in enumerate(distances_keys):
        predictions[i], predictions_err[i] = dictionary[key]
    result = np.zeros((len(distances), 3))
    for i in np.arange(len(distances)):
        result[i, 0] = 1 / distances[i]
        # print(predictions[i])
        # result[i, 1] = cls.find_crossing(noises, predictions[i])
        result[i, 1], result[i, 2] = get_crossing_bootstrap(noises, predictions[i, :, 0],
                                                            predictions_err[i, :, 0],
                                                            noises,
                                                            predictions[i, :, 1], predictions_err[i, :, 1], 0,
                                                            0.3)
    return result


def simple_bootstrap(x, f=np.mean, c=0.68, r=100):
    """ Use bootstrap resampling to estimate a statistic and
    its uncertainty.

    x (1d array): the data
    f (function): the statistic of the data we want to compute
    c (float): confidence interval in [0, 1]
    r (int): number of bootstrap resamplings

    Returns estimate of stat, upper error bar, lower error bar.
    """
    assert 0 <= c <= 1, 'Confidence interval must be in [0, 1].'
    # number of samples
    n = len(x)

    # stats of resampled datasets
    fs = np.asarray(
        [f(x[np.random.randint(0, n, size=n)]) for _ in range(r)]
    )
    # estimate and upper and lower limits
    med = 50  # median of the data
    val = np.percentile(fs, med)
    high = np.percentile(fs, med * (1 + c))
    low = np.percentile(fs, med * (1 - c))
    # estimate and uncertainties
    return val, high - val, val - low


def get_spline(x, y, err):
    idxs = np.argsort(x)
    err = np.array(list(map(lambda z: max(z, 1e-5), err)))
    spl = UnivariateSpline(x[idxs], y[idxs], w=1. / err[idxs])  # Smoothing spline, respects weights of data points!
    return spl


def get_crossing_mean(x1, y1, err1, x2, y2, err2, xmin, xmax):
    spl1 = get_spline(x1, y1, err1)
    spl2 = get_spline(x2, y2, err2)

    def diff(x):
        return spl1(x) - spl2(x)

    try:
        xcross = brentq(diff, xmin, xmax)
        return xcross
    except Exception:
        return float('NaN')


def get_crossing_bootstrap(x1, y1, err1, x2, y2, err2, xmin, xmax):
    xcross = get_crossing_mean(x1, y1, err1, x2, y2, err2, xmin, xmax)

    nsamples = 500

    samples = []
    for n in range(nsamples):
        x_sample = get_crossing_mean(x1, np.random.normal(loc=y1, scale=err1), err1, x2,
                                     np.random.normal(loc=y2, scale=err2), err2, xmin, xmax)
        if np.isfinite(x_sample):
            samples.append(x_sample)

    c = 0.68

    xcross_err = np.std(samples)
    return xcross, xcross_err
