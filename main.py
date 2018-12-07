import pickle
import numpy as np
from PyDDE import pydde
import matplotlib.pyplot as plt
import utils
import copy
import scipy.signal


def plot_time_series(result):
    tt, x1, x2, theta = zip(*result)
    t = np.array(tt) / 1000
    plt.figure()
    plt.plot(t, x1)
    plt.plot(t, x2)
    plt.plot(t, theta)
    plt.xlabel('Time [s]')
    plt.ylabel('Activity [spk/s]')
    plt.legend(['STN', 'GPe', 'Theta'])
    plt.show()


def plot_spectral_density(result, fs):
    """Plot spectral density estimated with Welch method

    :param result:
    :return:
    """
    tt, x1, x2, theta = zip(*result)
    f1, density1 = scipy.signal.welch(np.array(x1), fs)
    f2, density2 = scipy.signal.welch(np.array(x2), fs)
    dominant_f1 = f1[np.argmax(density1)]
    dominant_f2 = f2[np.argmax(density2)]
    print('Dominant frequency STN=%2.2f Hz, GPe=%2.2f Hz' % (dominant_f1, dominant_f2))
    plt.figure()
    plt.semilogy(f1, density1)
    plt.semilogy(f2, density2)
    plt.legend(['STN', 'GPe'])
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('PSD')
    plt.show()


def save_simulation_results(params, result):
    # TODO: saving results
    pass


def run_simulation(params):
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

    s1 = utils.create_activation_function(params['activation_function_1'])
    s2 = utils.create_activation_function(params['activation_function_2'])

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
        'tstop': 5000,
        'dt': 0.3,
        'activation_function_1': {
            'type': 'saturation',
            'min': 16,
            'max': 300,
            'slope': 0.7
        },
        'activation_function_2': {
            'type': 'saturation',
            'min': 32,
            'max': 400,
            'slope': 0.45
        }
    }

    simulation_data = run_simulation(params)
    save_simulation_results(params, simulation_data)
    plot_time_series(simulation_data)
    fs = 1000 / params['dt']
    plot_spectral_density(simulation_data, fs)
