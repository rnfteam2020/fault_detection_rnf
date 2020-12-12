"""
Data and Dataset generation
"""
import numpy as np
import scipy as sp
from scipy.integrate import odeint
import matplotlib.pyplot as plt


def u_sin(t):
    """
    Sine function input signal

    :param t: time

    :return: vector
    """
    return np.sin(t)

def u_step(t):
    """
    Step function as input signal

    :param t: time

    :return: vector
    """
    return 0 if t < 20 else 1

def model(x, t, b, k, m):
    """
    ODE model: spring, mass damp.

    :param x: input x value
    :param t: time value
    :param b: damping
    :param k: spring stiffness
    :param m: mass

    :return dx: derivative x
    """
    dx0 = x[1]
    dx1 = -k/m*x[0] - b/m*x[1] + u_step(t)/m
    dx = [dx0, dx1]

    return dx

def test_model():
    """
    Test model function

    if we pass k,b,m = 0 => no spring, damp, mass in system =>  achieve
    some fault.
    """
    n = 5001
    t = np.linspace(0, 50, n)

    x0 = [0.1, 0]
    b = 1
    k = 100
    m = 1

    y = odeint(model, x0, t, args=(b, k, m))
    plt.plot(t, y)
    plt.show()


def generate_data_from_model(b=1, k=100, m=1, n=5001, t_max=50):
    """
    Dataset generator with parameters

    :param n: number of samples
    :param t_max: max value of t vector
    :param b: damping
    :param k: spring stiffness
    :param m: mass

    :return t: time vector
    :return u: input to model vector
    :return y: model response

    """
    n = 5001
    t = np.linspace(0, 50, n)

    x0 = [0, 0]
    b = 1
    k = 100
    m = 1

    y = odeint(model, x0, t, args=(b, k, m))
    u = np.asarray(list(map(u_step, t)), dtype="float")    # control signal to model
    y = y

    return [t, u, y]

def generate_signals_with_labels():
    # TODO
    """
    Generate health and fault data

    :return data: data = {'label': 0/1, 'signals': np.array([t,u,y])}

    """
    m_max = 500
    m_min = 10
    m_N = 10
    m_arr = np.linspace(m_min, m_max, m_N)

    k_max = 5000
    k_min = 10
    k_N = m_N
    k_arr = np.linspace(k_min, k_max, k_N)[::-1]

    b_max = 2000
    b_min = 10
    b_N = m_N
    b_arr = np.linspace(b_min, b_max, b_N)

    # podminka 0
    # pokud ma soustava pomerny utlum >= 1 je pretlumena a nedochazi ke kmitani
    # pomerny utlum
    b_pom = b_arr / (2 * np.sqrt(m_arr * k_arr))

    labels = [1 if x < 1 else 0 for x in b_pom]

    dat = list(map(generate_data_from_model, b_arr, k_arr, m_arr))

    # zip doesnt work here
    #data = dict(zip(labels, dat))

    data = []
    for i in range(len(labels)):
        data.append({labels[i]:dat[i]})

    return data


def verification(net, u):
    # TODO
    pass

if __name__ == "__main__":
    data = generate_signals_with_labels()
    for d in data:
        for label, signal in d.items():
            print(f'{label} : {signal}')

