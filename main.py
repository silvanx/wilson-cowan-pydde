import pickle
import numpy as np
from PyDDE import pydde
import matplotlib.pyplot as plt
import utils


def ddegrad(s, c, t):
    """Equations defining time evolution of the system. Returns the value of the derivative at time t

    :param s: state variables
    :param c: constants
    :param t: time
    :return:
    """

    initial_state = [c[16], c[17], c[18]]

    max_delay = max(c[6:10])
    if t > max_delay:
        delayed_values = [
            pydde.pastvalue(0, t - c[6], 0),  # x1d11
            pydde.pastvalue(1, t - c[7], 1),  # x2d12
            pydde.pastvalue(0, t - c[8], 2),  # x1d21
            pydde.pastvalue(1, t - c[9], 3)   # x2d22
        ]
    else:
        delayed_values = [
            initial_state[0],
            initial_state[1],
            initial_state[0],
            initial_state[1]
        ]

    inputs = [
        c[2] * delayed_values[0] - c[3] * delayed_values[1] + c[14] - s[0] * s[2],
        c[4] * delayed_values[2] - c[5] * delayed_values[3] - c[15]
    ]

    theta_dot = 0
    return np.array([
        1/c[0] * (-s[0] + s1(inputs[0])),
        1/c[1] * (-s[1] + s2(inputs[1])),
        theta_dot
    ])


def plot_simulation_results(tt, x1, x2, theta):
    plt.figure()
    plt.plot(tt, x1)
    plt.plot(tt, x2)
    plt.plot(tt, theta)
    plt.legend(['STN', 'GPe', 'Theta'])
    plt.show()


def save_simulation_results(params, tt, x1, x2, theta):
    # TODO: saving results
    pass


def run_simulation(params):
    system = pydde.dde()
    # TODO: make the constants table out of params
    constants = [
        6,          # 0:  tau1 [ms]
        14,         # 1:  tau2 [ms]
        0,          # 2:  c11
        20,         # 3:  c12
        30,       # 4:  c21
        5,          # 5:  c22  !
        0,          # 6:  d11
        6,          # 7:  d12 [ms]
        6,          # 8:  d21 [ms]
        4,          # 9:  d22 [ms]
        300,        # 10: m1 [spk/s]
        17,         # 11: b1 [spk/s]
        400,        # 12: m2 [spk/s]
        75,         # 13: b2 [spk/s]
        90 * 9.2,   # 14: u1 [spk/s] !
        9 * 139.4,  # 15: u2 [spk/s]
        40,         # 16: x1_0 [spk/s]
        40,         # 17: x2_0 [spk/s]
        0           # 18: theta_0
    ]
    # TODO: set dt and tstop based on params
    tstop = 3500
    dt = 0.5

    initial_state = np.array([constants[16], constants[17], constants[18]])

    system.dde(y=initial_state, times=np.arange(0.0, tstop, dt), func=ddegrad, parms=constants,
               tol=10e-7, dt=dt, nlag=4, hbsize=800000)

    tt, x1, x2, theta = zip(*system.data)
    return tt, x1, x2, theta


if __name__ == "__main__":
    print('Wilson-Cowan simulation')

    # TODO: read params from a config file
    params = {}

    # TODO: initialize activation functions based on params
    def s1(x):
        return utils.saturation(x, 16, 300, 0.5)


    def s2(x):
        return utils.saturation(x, 30, 400, 0.3)

    tt, x1, x2, theta = run_simulation(params)
    save_simulation_results(params, tt, x1, x2, theta)
    plot_simulation_results(tt, x1, x2, theta)
