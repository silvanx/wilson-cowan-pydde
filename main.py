import pickle
import numpy as np
from PyDDE import pydde
import matplotlib.pyplot as plt
import utils
import copy


def plot_simulation_results(result):
    tt, x1, x2, theta = zip(*result)
    plt.figure()
    plt.plot(tt, x1)
    plt.plot(tt, x2)
    plt.plot(tt, theta)
    plt.legend(['STN', 'GPe', 'Theta'])
    plt.show()


def save_simulation_results(params, result):
    # TODO: saving results
    pass


def run_simulation(params):

    # TODO: initialize activation functions based on params
    def s1(x):
        return utils.saturation(x, 16, 300, 0.5)

    def s2(x):
        return utils.saturation(x, 30, 400, 0.3)

    def ddegrad(s, c, t):
        """Equations defining time evolution of the system. Returns the value of the derivative at time t

        :param s: state variables
        :param c: constants
        :param t: time
        :return:
        """

        max_delay = max(c[6:10])
        if t > max_delay:
            delayed_values = [
                pydde.pastvalue(0, t - c[6], 0),  # x1d11
                pydde.pastvalue(1, t - c[7], 1),  # x2d12
                pydde.pastvalue(0, t - c[8], 2),  # x1d21
                pydde.pastvalue(1, t - c[9], 3)  # x2d22
            ]
        else:
            # initial_state taken from the outer scope
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
            1 / c[0] * (-s[0] + s1(inputs[0])),
            1 / c[1] * (-s[1] + s2(inputs[1])),
            theta_dot
        ])

    system = pydde.dde()
    constants = [
        params['tau1'],   # 0:  tau1 [ms]
        params['tau2'],   # 1:  tau2 [ms]
        params['c11'],    # 2:  c11
        params['c12'],    # 3:  c12
        params['c21'],    # 4:  c21
        params['c22'],    # 5:  c22  !
        params['d11'],    # 6:  d11
        params['d12'],    # 7:  d12 [ms]
        params['d21'],    # 8:  d21 [ms]
        params['d22'],    # 9:  d22 [ms]
        params['m1'],     # 10: m1 [spk/s]
        params['b1'],     # 11: b1 [spk/s]
        params['m2'],     # 12: m2 [spk/s]
        params['b2'],     # 13: b2 [spk/s]
        params['u1'],     # 14: u1 [spk/s] !
        params['u2'],     # 15: u2 [spk/s]
        params['x10'],    # 16: x1_0 [spk/s]
        params['x20'],    # 17: x2_0 [spk/s]
        params['theta0']  # 18: theta_0
    ]
    tstop = params['tstop']
    dt = params['dt']

    initial_state = np.array([constants[16], constants[17], constants[18]])

    system.dde(y=initial_state, times=np.arange(0.0, tstop, dt), func=ddegrad, parms=constants,
               tol=10e-7, dt=dt, nlag=4, hbsize=800000)

    return copy.copy(system.data)


if __name__ == "__main__":
    print('Wilson-Cowan simulation')

    # TODO: read params from a config file
    params = {
        'tau1': 6,
        'tau2': 14,
        'c11': 0,
        'c12': 20,
        'c21': 30,
        'c22': 5,
        'd11': 0,
        'd12': 6,
        'd21': 6,
        'd22': 4,
        'm1': 300,
        'b1': 17,
        'm2': 400,
        'b2': 75,
        'u1': 90 * 9.2,
        'u2': 9 * 139.4,
        'x10': 40,
        'x20': 40,
        'theta0': 0,
        'tstop': 3500,
        'dt': 0.5
    }

    simulation_data = run_simulation(params)
    save_simulation_results(params, simulation_data)
    plot_simulation_results(simulation_data)
