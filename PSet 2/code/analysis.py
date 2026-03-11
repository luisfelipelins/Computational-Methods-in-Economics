# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 10:06:00 2026

@author: lfval
"""

import timeit
import functions
import numpy as np
import pandas as pd
from config import OUTPUTS
import matplotlib.pyplot as plt
from scipy import interpolate as itplt


if __name__ == '__main__':
    
    ### --- Question 1 --- ###
    x = np.array([1940,1950,1960,1970,1980,1991,2000,2010])
    y = np.array([45.5,48.0,52.5,57.6,62.5,66.9,71.1,74.4])
    
    years           : range          = range(1940,2011)
    q1_interpolation: itplt.interp1d = itplt.interp1d(x=x, y=y,kind='quadratic')
    q1_interpolation: np.ndarray     = q1_interpolation(years)
    
    fig,ax = plt.subplots(figsize=(9,5))
    ax.grid(linestyle='--',zorder=0)
    ax.scatter(x,y,zorder=5,s=30,color='red',label='Actual Points')
    ax.plot(years,q1_interpolation,color='black',label='Interpolation')
    ax.text(0.70, 0.05, 
            f"Life Expectancy in 1996: {round(pd.Series(index=years,data=q1_interpolation)[1996],1)}", 
            transform=ax.transAxes, fontweight='bold', fontsize=10, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white'))
    ax.legend(loc=0)
    ax.set_xlabel('Year')
    ax.set_ylabel('Life Expectancy')
    plt.tight_layout()
    
    plt.savefig(OUTPUTS / 'chart_q1.pdf')
    plt.close()
    
    
    
    ### --- Question 2 --- ###
    
    # Item d)
    functions.interpolation_wrapper(func=functions.f_2,sample_n=2500,int_lb=1,int_ub=5,name='q2_d',log_grid=False)
    
    # Item e)
    functions.interpolation_wrapper(func=functions.f_2,sample_n=2500,int_lb=1,int_ub=5,name='q2_e',log_grid=True)
    
    
    
    ### --- Question 3 --- ###
    
    # Normal scale
    functions.interpolation_wrapper(func=functions.g,sample_n=2500,int_lb=200,int_ub=1000,name='q3_normal',log_grid=False)
    
    # Log scale
    functions.interpolation_wrapper(func=functions.g,sample_n=2500,int_lb=200,int_ub=1000,name='q3_log',log_grid=True)
    
    
    
    ### --- Question 4 --- ###
    
    # Item a)
    a = functions.f_4(-2)
    b = functions.f_4(2)
    
    print(f'f(-2)={a}')
    print(f'f(2)={b}')
    
    # Item b)
    bisection_res: tuple = functions.bisection(func      = functions.f_4,
                                               fixed_val = 0,
                                               a         = -2,
                                               b         = 2,
                                               eps       = 1e-8)
    
    # Item c)
    secant_res: tuple = functions.secant(func      = functions.f_4,
                                         x_0       = np.array([-2]),
                                         fixed_val = np.array([0]),
                                         h         = 1e-5,
                                         eps       = 1e-8)
    
    # Item d)
    nr_res: tuple = functions.newton_rapthon(func      = functions.f_4,
                                             jacobian  = functions.f_4_prime,
                                             x_0       = np.array([-2]),
                                             fixed_val = np.array([0]),
                                             eps       = 1e-8)
    
    print(f"Bisection Result: value = {round(bisection_res[0],4)} | time =  {round(bisection_res[1],4)}s | iterations =  {bisection_res[2]}")
    print(f"Secant Result: value = {round(secant_res[0][0],4)} | time =  {round(secant_res[1],4)}s | iterations =  {secant_res[2]}")
    print(f"Newton-Raphson Result: value = {round(nr_res[0][0],4)} | time =  {round(nr_res[1],4)}s | iterations =  {nr_res[2]}")






