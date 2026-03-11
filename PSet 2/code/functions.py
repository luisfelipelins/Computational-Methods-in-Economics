# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 10:02:16 2026

@author: lfval
"""

import time
import timeit
import math
import numpy as np
import pandas as pd
from scipy import interpolate
import matplotlib.pyplot as plt
from config import OUTPUTS

def bisection(func,fixed_val,a,b,eps,print_convergence=True,**kwargs):
    """
    Solve an equations using the Bisection method.

    Parameters
    ----------
    func : callable
        Univariate function that takes a float and returns a float. Represents 
        the equations to be solved.
    fixed_val : float
        A float with the fixed values trying to be recovered. For the root,
        set it to 0.
    a : float
        Lower bound of the initial interval.
    b : float
        Upper bound of the initial interval.
    eps : float
        Error tolerance for the bisection convergence.
    **kwargs
        Additional arguments passed to `func`, such as parameters.

    Returns
    -------
    c : float
        The fixed value/root numerically estimated by bisection.
    end: float
        The time in seconds elapsed for convergence.
    it_count: int
        The number of iterations to convergence.
    

    """
    
    start = time.time()
    
    c = (a+b)/2
    
    error_abs = abs(func(c,**kwargs) - fixed_val)
    
    it_count = 0
    while error_abs > eps:
        it_count += 1
        c = (a+b)/2
        
        error_abs = abs(func(c,**kwargs) - fixed_val)
        error     = func(c,**kwargs) - fixed_val
        
        if error > 0:
            b = c
        else:
            a = c
    
    end = time.time() - start
    
    if print_convergence: print(f'Convergence in {end} seconds, after {it_count} iterations.')
    
    return c, end, it_count

def newton_rapthon(func,jacobian,x_0,fixed_val,eps,timeout=1000,print_convergence=True,**kwargs):
    """
    Solve a system of equations using the Newton–Raphson method.

    Parameters
    ----------
    func : callable
        Function that takes a 1D NumPy array of shape (n,) and returns
        a 1D NumPy array of shape (n,). Represents the system of
        equations to be solved.
    jacobian : callable
        Function that takes a 1D NumPy array of shape (n,) and returns
        the Jacobian matrix of shape (n, n).
    x_0 : numpy.ndarray
        Initial guess for the solution, 1D array of shape (n,).
    fixed_val : numpy.ndarray
        A 1D NumPy array of shape (n,) with the fixed values trying to be 
        recovered. For the root, set it to 0.
    eps : float
        Error tolerance for each entry of the 1D NumPy array of shape (n,)
        'fixed_val' array.
    timeout : int, optional
        Maximum number of iterations before raising a TimeoutError.
        Default is 1000.
    **kwargs
        Additional arguments passed to `func` and `jacobian`, such as parameters.

    Returns
    -------
    numpy.ndarray
        Estimated fixed value solution `x`.
    """
    
    start = time.time()
    it_count = 0
    
    f_x_0 = func(x_0,**kwargs)
    error = abs(f_x_0 - fixed_val)
    cond = any([i>eps for i in error])
    
    prev_x = x_0
    timeout_error = False

    while cond:
        it_count += 1
        
        if np.shape(jacobian(prev_x,**kwargs)) == (1,):
            next_x   = prev_x - np.dot(np.array([1/jacobian(prev_x,**kwargs)], dtype=float),func(prev_x,**kwargs))
        else:
            next_x   = prev_x - np.dot(np.linalg.inv(jacobian(prev_x,**kwargs)),func(prev_x,**kwargs))
            
        next_f_x = func(next_x,**kwargs)
        
        if not any([math.isfinite(i) for i in next_f_x]):
            prev_x = (prev_x+next_x)/2
        else:
            error = abs(next_f_x - fixed_val)
            cond = any([i>eps for i in error])
            prev_x = next_x.copy()
        
        if it_count > timeout:
            if print_convergence: print("Model couldn't converge!")
            timeout_error = True
            break
    
    end = time.time() - start

    if timeout_error:
        ret = (np.array([np.nan], dtype=float), end, it_count)
    else:
        ret = (next_x, end, it_count)
        if print_convergence: print(f'Convergence in {end} seconds, after {it_count} iterations.')
    
    return ret

def secant(func,x_0,fixed_val,h,eps,timeout=1000,print_convergence=True,**kwargs):
    """
    Solve a system of equations using the secant method.

    Parameters
    ----------
    func : callable
        Function that takes a 1D NumPy array of shape (n,) and returns
        a 1D NumPy array of shape (n,). Represents the system of
        equations to be solved.
    x_0 : numpy.ndarray
        Initial guess for the solution, 1D array of shape (n,).
    fixed_val : numpy.ndarray
        A 1D NumPy array of shape (n,) with the fixed values trying to be 
        recovered. For the root, set it to 0.
    h : float
        Derivative approximation size.
    eps : float
        Error tolerance for each entry of the 1D NumPy array of shape (n,)
        'fixed_val' array.
    timeout : int, optional
        Maximum number of iterations before raising a TimeoutError.
        Default is 1000.
    **kwargs
        Additional arguments passed to `func` and `jacobian`, such as parameters.

    Returns
    -------
    numpy.ndarray
        Estimated fixed value solution `x`.
    """
    
    start = time.time()
    it_count = 0
    
    f_x_0 = func(x_0,**kwargs)
    error = abs(f_x_0 - fixed_val)
    cond = any([i>eps for i in error])
    
    prev_x = x_0
    timeout_error = False
    
    n = x_0.shape[0]

    while cond:
        it_count += 1
        
        jacobian = np.full((n, n), np.nan, dtype=float)
        
        for i in range(0,n):
            temp_x = prev_x.copy()
            temp_x[i] = temp_x[i] + h
            temp_f = np.array([i/h for i in func(temp_x,**kwargs) - func(prev_x,**kwargs)],dtype=float)
            jacobian[:,i] = temp_f

        if jacobian.shape == (1,):
            next_x   = prev_x - np.dot(np.array([1/jacobian], dtype=float),func(prev_x,**kwargs))
        else:
            next_x   = prev_x - np.dot(np.linalg.inv(jacobian),func(prev_x,**kwargs))
            
        next_f_x = func(next_x,**kwargs)
        
        if not any([math.isfinite(i) for i in next_f_x]):
            prev_x = (prev_x+next_x)/2
        else:
            error = abs(next_f_x - fixed_val)
            cond = any([i>eps for i in error])
            prev_x = next_x.copy()
        
        if it_count > timeout:
            if print_convergence: print("Model couldn't converge!")
            timeout_error = True
            break
    
    end = time.time() - start

    if timeout_error:
        ret = (np.array([np.nan], dtype=float), end, it_count)
    else:
        ret = (next_x, end, it_count)
        if print_convergence: print(f'Model converged after {end}s and {it_count} iterations.')
    
    return ret

def f_2(x,x_m=1,α=10):
    '''
    Returns the value of the function f(x) defined in Question 2

    Parameters
    ----------
    x : float
        Input of f(x).
    x_m : float
        Parameter x_m. The default is 1.
    α : float
        Parameter α. The default is 10.

    Returns
    -------
    f : float
        Value of f(x).

    '''
    
    f: float = (α*x_m**α)/(x**(α+1))
    
    return f

def g(x,η=5,ξ=500):
    '''
    Returns the value of the function g(x) defined in Question 3

    Parameters
    ----------
    x : float
        Input of g(x).
    η : float, optional
        Parameter η. The default is 5.
    ξ : float, optional
        Parameter ξ. The default is 500.

    Returns
    -------
    g : float
        Value of g(x).

    '''
    
    g: float = 1/(1+np.exp(-η*np.log(x/ξ)))
    
    return g

def f_4(x):
    '''
    Returns the value of the function f(x) defined in Question 4

    Parameters
    ----------
    x : float
        Input of f(x).

    Returns
    -------
    f : float
        Value of f(x).

    '''
    
    f: float = x**3+2*x+5

    return f

def f_4_prime(x):
    '''
    Returns the derivative of function f(x) defined in Question 4, to be used in
    the Newton-Raphson method.

    Parameters
    ----------
    x : float
        Input of f'(x).

    Returns
    -------
    f : float
        Value of f'(x).

    '''
    
    f_prime: float = (3*(x**2))+2
    
    return f_prime
    

def calc_interpolation_error(func,sample_n,obs_n,int_lb,int_ub,log_grid=False,timing_reps=200):
    '''
    Does the analysis described in Question 2, item c) for any function func

    Parameters
    ----------
    func : callable
        Function to be interpolated.
    sample_n : int
        Size of the true grid of the function.
    obs_n : int
        Size of the observation sample to be interpolated.
    int_lb : int
        Lower bound of the interval.
    int_ub : int
        Upper bound of the interval.
    log_grid : bool, optional
        Boolean to calculate the log grid. The default is False.
    timing_reps : int, optinal
        Number of times timeit repeats the evaluation call to compute average execution time. The default is 200.

    Returns
    -------
    ret : dict
        Dictionary with the results of the interpolation.

    '''
    
    rng      : np.random.Generator = np.random.default_rng(seed=42)
    sample_x : np.ndarray          = rng.uniform(int_lb, int_ub, sample_n)
    obs_x   : np.ndarray = (np.geomspace(int_lb, int_ub, obs_n) if log_grid else np.linspace(int_lb, int_ub, obs_n)) 

    sample_y: np.ndarray = func(sample_x) 
    obs_y   : np.ndarray = func(obs_x)

    # --- Linear ---
    linear   = interpolate.interp1d(x=obs_x, y=obs_y)
    linear_y = linear(sample_x)
    linear_eps  = ((linear_y - sample_y) ** 2).mean()
    linear_time = timeit.timeit(lambda: linear(sample_x), number=timing_reps) / timing_reps

    # --- Cubic Spline ---
    spline   = interpolate.CubicSpline(x=obs_x, y=obs_y)
    spline_y = spline(sample_x)
    spline_eps  = ((spline_y - sample_y) ** 2).mean()
    spline_time = timeit.timeit(lambda: spline(sample_x), number=timing_reps) / timing_reps

    # --- Akima ---
    akima   = interpolate.Akima1DInterpolator(x=obs_x, y=obs_y)
    akima_y = akima(sample_x)
    akima_eps  = ((akima_y - sample_y) ** 2).mean()
    akima_time = timeit.timeit(lambda: akima(sample_x), number=timing_reps) / timing_reps

    ret: dict = {'Linear ε' : linear_eps,
                 'Spline ε' : spline_eps,
                 'Akima ε'  : akima_eps,
                 'Linear Δ' : linear_time,
                 'Spline Δ' : spline_time,
                 'Akima Δ'  : akima_time}
    
    return ret

def interpolation_wrapper(func,sample_n,int_lb,int_ub,name,log_grid=False):
    '''
    Wrapper to solve interpolation questions in Questions 2 and 3

    Parameters
    ----------
    func : callable
        Function to be interpolated.
    sample_n : int
        Size of the true grid of the function.
    int_lb : int
        Lower bound of the interval.
    int_ub : int
        Upper bound of the interval.
    name : str
        String with the name to be appended to the resulting tables and charts.
    log_grid : bool, optional
        Boolean to calculate the log grid. The default is False.

    Returns
    -------
    None.

    '''
    
    res_list: list = []
    for n in [10,15,20,30,50]:
        res: dict = calc_interpolation_error(func=func, sample_n=sample_n, obs_n=n, int_lb=int_lb, int_ub=int_ub, log_grid=log_grid)
        res_list.append(pd.Series(res,name=f'n={n}'))

    res: pd.DataFrame = pd.concat(res_list,axis=1)
    
    res.to_excel(OUTPUTS / f'table_{name}.xlsx')

    ns     : list = [10, 15, 20, 30, 50]
    methods: dict = {'Linear': {'eps': 'Linear ε', 'time': 'Linear Δ', 'color': '#5B8CFF'},
                     'Spline': {'eps': 'Spline ε', 'time': 'Spline Δ', 'color': '#CC5BFF'},
                     'Akima' : {'eps': 'Akima ε',  'time': 'Akima Δ',  'color': '#FF5B8C'}}

    fig, ax = plt.subplots(figsize=(9, 5))

    for label, cfg in methods.items():
        x = res.loc[cfg['time']]
        y = res.loc[cfg['eps']]

        ax.plot(x, y, color=cfg['color'], linewidth=2, zorder=2, label=label)
        ax.scatter(x, y, color=cfg['color'], s=500, zorder=3)

        for xi, yi, n in zip(x, y, ns):
            ax.text(xi, yi, str(n), color='white',
                    fontsize=7, fontweight='bold',
                    ha='center', va='center', zorder=4)

    ax.set_xlabel('Avg. Evaluation Time — Δ (s)', fontsize=11)
    ax.set_ylabel('Mean Squared Error — ε', fontsize=11)
    ax.legend(framealpha=0.3)
    ax.grid(True, linestyle='--', alpha=0.4)

    plt.tight_layout()

    plt.savefig(OUTPUTS / f'chart_{name}.pdf')
    plt.close()