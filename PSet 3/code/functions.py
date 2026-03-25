# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 10:00:00 2026

@author: lfval
"""

import numpy as np
import matplotlib.pyplot as plt
from config import OUTPUTS


def f_q1(x):
    '''
    Function f from question 1

    Parameters
    ----------
    x : float or np.ndarray
        Input value(s).

    Returns
    -------
    float or np.ndarray
        Value of f(x).
    '''
    return x * np.sin(5 * x)


def f_q2(x1, x2, theta):
    '''
    Function f from question 2.
    
    Parameters
    ----------
    x1 : float
        First input variable.
    x2 : float
        Second input variable.
    theta : array-like of length 4
        Parameter vector (θ_1, θ_2, θ_3, θ_4).

    Returns
    -------
    float
        Value of f(x1, x2; θ).
    '''
    θ_1, θ_2, θ_3, θ_4 = theta
    return θ_1 * x1 + θ_2 * np.exp(-x2**2) + θ_3 * np.log(1 + np.abs(x2)) + θ_4 * x1**x2


def g_q2(theta):
    '''
    Sum of squared residuals g(θ) between the model predictions and the four observed values.

    Parameters
    ----------
    theta : array-like of length 4
        Parameter vector (θ_1, θ_2, θ_3, θ_4).

    Returns
    -------
    float
        Value of g(θ).
    '''
    
    y_obs   = [43.614, 563.694, 43.230, 23.130]
    x_vals  = [(1, 1), (2, 4), (-1, 2), (2, -2)]

    g_val = 0.0
    for (x1, x2), y in zip(x_vals, y_obs):
        g_val += (f_q2(x1, x2, theta) - y) ** 2

    return g_val


class CallbackRecorder:
    '''
    Callback object to record iteration history of scipy.optimize.minimize.

    Attributes
    ----------
    thetas : list
        List of parameter vectors at each iteration.
    g_vals : list
        List of g(θ) values at each iteration.
    '''

    def __init__(self):
        self.thetas = []
        self.g_vals = []

    def __call__(self, xk):
        self.thetas.append(xk.copy())
        self.g_vals.append(g_q2(xk))


def plot_q2_convergence(recorder):
    '''
    Plots the convergence of the optimization in Question 2.

    Parameters
    ----------
    recorder : CallbackRecorder
        Object with recorded iteration history.
    '''
    iterations = range(1, len(recorder.g_vals) + 1)
    thetas_arr = np.array(recorder.thetas)

    # Plot g(θ) vs iteration
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.grid(linestyle='--', zorder=0)
    ax.plot(iterations, recorder.g_vals, color='black', linewidth=1.5, zorder=2)
    ax.scatter(iterations, recorder.g_vals, color='red', s=15, zorder=3)
    ax.set_xlabel('Iteration')
    ax.set_ylabel(r'$g(\theta_i)$')
    ax.set_title(r'Convergence of $g(\theta)$')
    plt.tight_layout()
    plt.savefig(OUTPUTS / 'chart_q2_convergence.pdf')
    plt.close()

    # Plot each θ component vs iteration
    labels = [r'$\theta_1$', r'$\theta_2$', r'$\theta_3$', r'$\theta_4$']
    for j in range(4):
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.grid(linestyle='--', zorder=0)
        ax.plot(iterations, thetas_arr[:, j], color='black', linewidth=1.5, zorder=2)
        ax.scatter(iterations, thetas_arr[:, j], color='red', s=15, zorder=3)
        ax.set_xlabel('Iteration')
        ax.set_ylabel(labels[j])
        ax.set_title(f'Parameter {labels[j]} across iterations')
        plt.tight_layout()
        plt.savefig(OUTPUTS / f'chart_q2_theta{j+1}.pdf')
        plt.close()


def gauss_hermite_expected_max(n_nodes=20):
    '''
    Computes E[max{X, Y}] using Gauss-Hermite quadrature for X, Y ~ N(0,1).

    The substitution x = sqrt(2) * t transforms the standard normal density
    into the Gauss-Hermite weight function exp(-t^2) / sqrt(π).

    Parameters
    ----------
    n_nodes : int, optional
        Number of quadrature nodes per dimension. The default is 20.

    Returns
    -------
    float
        Estimated E[max{X, Y}].
    '''
    nodes, weights = np.polynomial.hermite.hermgauss(n_nodes)

    # Transform nodes: x = sqrt(2) * t
    x = np.sqrt(2) * nodes
    w = weights / np.sqrt(np.pi)

    result = 0.0
    for i in range(n_nodes):
        for j in range(n_nodes):
            result += w[i] * w[j] * np.maximum(x[i], x[j])

    return result


def monte_carlo_expected_max(n_draws=1000000, seed=13051905):
    '''
    Computes E[max{X, Y}] using Monte Carlo integration for X, Y ~ N(0,1).

    Parameters
    ----------
    n_draws : int, optional
        Number of draws. The default is 1,000,000.
    seed : int, optional
        Random seed. The default is 13051905.

    Returns
    -------
    float
        Estimated E[max{X, Y}].
    '''
    rng = np.random.default_rng(seed=seed)
    x = rng.standard_normal(n_draws)
    y = rng.standard_normal(n_draws)

    return np.mean(np.maximum(x, y))

def trapezoid(func, a, b, n):
    '''
    Computes the integral of func over [a, b] using the trapezoid rule with n nodes.

    Parameters
    ----------
    func : callable
        Function to integrate.
    a : float
        Lower bound.
    b : float
        Upper bound.
    n : int
        Number of nodes (including endpoints).

    Returns
    -------
    float
        Estimated integral value.
    '''
    x = np.linspace(a, b, n)
    y = func(x)
    h = (b - a) / (n - 1)

    return h * (0.5 * y[0] + np.sum(y[1:-1]) + 0.5 * y[-1])


def f_q4a(x):
    value = x
    return value

def f_q4b(x):
    value = x * np.sin(x)
    return value

def f_q4c(x):
    value = np.sqrt(1 - x**2)
    return value


def centered_diff_2pt(func, x, h):
    '''
    Computes the numerical derivative using 2-point centered differences.

    f'(x) = [f(x+h) - f(x-h)] / (2h)

    Parameters
    ----------
    func : callable
        Function to differentiate.
    x : float
        Point at which to compute the derivative.
    h : float
        Step size.

    Returns
    -------
    float
        Estimated derivative.
    '''
    return (func(x + h) - func(x - h)) / (2 * h)


def centered_diff_4pt(func, x, h):
    '''
    Computes the numerical derivative using 4-point centered differences.

    f'(x) = [-f(x+2h) + 8f(x+h) - 8f(x-h) + f(x-2h)] / (12h)

    Parameters
    ----------
    func : callable
        Function to differentiate.
    x : float
        Point at which to compute the derivative.
    h : float
        Step size.

    Returns
    -------
    float
        Estimated derivative.
    '''
    
    return (-func(x + 2*h) + 8*func(x + h) - 8*func(x - h) + func(x - 2*h)) / (12 * h)


def f_q5a(x):
    value = x**2
    return value

def f_q5a_prime(x):
    value = 2*x
    return value

def f_q5b(x):
    value = np.log(x)
    return value

def f_q5b_prime(x):
    value = 1/x
    return value

def f_q5c(x):
    value = x * np.sin(x)
    return value

def f_q5c_prime(x):
    value = np.sin(x) + x * np.cos(x)
    return value
