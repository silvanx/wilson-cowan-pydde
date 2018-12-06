import numpy as np


def sigmoid(x, m, b):
    """Sigmoidal function
    :param x: input
    :param m: maximum value of the sigmoid
    :param b: value at 0
    :return:
    """
    y = m / (1 + np.exp(-4 * x / m) * (m - b) / b)
    return y


def saturation(x, ymin, ymax, slope):
    """Piecewise linear function, constant below 0
    :param x: input argument
    :param ymin: value for x < 0
    :param ymax: maximum value
    :param slope: slope of the linear part
    :return:
    """
    maxpoint = (ymax - ymin) / slope
    y = np.piecewise(
        x,
        [x < 0,
         np.logical_and(x >= 0, x <= maxpoint),
         x > maxpoint],
        [ymin,
         ymin + x * slope,
         ymax]
    )
    return y
