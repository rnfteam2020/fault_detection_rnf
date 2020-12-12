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


def generate_data_from_model(n=5001, t_max=50, b=1, k=100, m=1):
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
    y = y[:,0]

    return t, u, y

def verification(net, u):
    # TODO
    pass

if __name__ == "__main__":
    pass
