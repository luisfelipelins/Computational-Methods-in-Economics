# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 08:26:18 2026

@author: lfval
"""

import numpy as np
import pandas as pd
from config import DATA_FINAL, OUTPUTS
import matplotlib.pyplot as plt
from statsmodels.nonparametric.kernel_regression import KernelReg

# Defining functions

def lin_regression(df,y,x,log_vars,const=True):
    '''
    Finds the coefficients of a multivariate linear regression

    Parameters
    ----------
    df : pandas.DataFrame
        Data Frame with the dataset.
    y : list
        List containing the name of the endogenous variables.
    x : list
        List containing the names of the exogenous variables.
    log_vars : list
        List containing the name of the variables to be considered in ln.
    const : boolean, optional
        Boolean to include a constant in the regression. The default is True.

    Returns
    -------
    b : dict
        Dictionary containing the betas estimated in the regression.

    '''
    
    # Preparing data
    temp_df: pd.DataFrame = df.copy()
    
    if const:
        temp_df['const'] = 1
        x                = ['const'] + x
    else: pass
    
    for c in log_vars:
        temp_df[c] = np.log(temp_df[c])
    
    y_val: np.ndarray = temp_df[y].values
    x_val: np.ndarray = temp_df[x].values
    
    # Running reg
    t1: np.ndarray = np.linalg.inv(x_val.T @ x_val)
    t2: np.ndarray = x_val.T @ y_val
    
    b: list  = list(t1 @ t2)
    b: dict  = {i:j[0] for i,j in zip(x,b)}
    
    return b

def plot_reg(df,reg_res,x_var,const=True):
    '''
    Creates an x and a y array to use to plot the regression line fit

    Parameters
    ----------
    df : pandas.DataFrame
        Data Frame with the dataset.
    reg_res : dict
        Dictionary with the regression results from lin_regression function.
    x_var : str
        String containing the name of the x variables under consideration.
        
    Returns
    -------
    plot_x : numpy.ndarray
        Array containing the x values to be used to plot the regression line.
    plot_y : numpy.ndarray
        Array containing the y values to be used to plot the regression line.

    '''
    
    min_x : float      = np.log(df[x_var]).min()
    max_x : float      = np.log(df[x_var]).max()
    plot_x: np.ndarray = np.linspace(start=min_x, stop=max_x,num=1000)
    
    if 'const' in list(reg_res.keys()):
        plot_y: np.ndarray = reg_res['const'] + reg_res[x_var]*plot_x
    else:
        plot_y: np.ndarray = reg_res[x_var]*plot_x
        
    return plot_x,plot_y

def item_a_reg_wrapper(df,y_var):
    '''
    Wrapper function for item (a) running the regression and creating the x and
    y vectors used to plot the line

    Parameters
    ----------
    y_var : str
        Name of the y var under consideration.

    Returns
    -------
    dict
        Dictionary with the estimated betas and the x and y vectors for plot.

    '''
    
    y            : list = [y_var]
    x            : list = ['prize_est','virada_dummy']
    log_vars     : list = [y_var,'prize_est']
    beta         : dict = lin_regression(df=df,y=y,x=x,log_vars=log_vars)
    
    plot_x,plot_y       = plot_reg(df=df,reg_res=beta,x_var='prize_est')
    
    return {'beta'   :beta,
            'plot_x' :plot_x,
            'plot_y' :plot_y}

def item_a(data):
    '''
    Performs the analysis requested in item (a), and saves the outputs in the file

    Parameters
    ----------
    data : pandas.DataFrame
        Data Frame with the dataset.

    Returns
    -------
    None.

    '''
    
    df: pd.DataFrame = data.copy()
    
    # Calculating correlations
    cor_cols: list = ['prize_est','revenue','ticket_count']
    cor_tab : pd.DataFrame = df[cor_cols].corr()
    
    cor_tab.to_csv(OUTPUTS / 'cor_tab_item_a.csv')
        
    # Estimating elasticities with revenue and ticket count
    revenue_reg  : dict = item_a_reg_wrapper(df,'revenue')
    tik_count_reg: dict = item_a_reg_wrapper(df,'ticket_count')
    
    # Plotting prize announcement vs revenue and ticket count
    fig,ax = plt.subplots(ncols=2,figsize=(9,5))

    ax[0].grid(linestyle='--',zorder=0)
    ax[0].scatter(np.log(df['prize_est']),np.log(df['revenue']),zorder=5,s=4,color='gray')
    ax[0].plot(revenue_reg['plot_x'],revenue_reg['plot_y'],color='red',zorder=10)
    ax[0].set_xlabel(r'$\ln$(Prize Estimate)')
    ax[0].set_ylabel(r'$\ln$(Revenue)')
    ax[0].text(0.05, 0.95, f"Elasticity: {round(revenue_reg['beta']['prize_est'],2)}", transform=ax[0].transAxes, fontweight='bold', fontsize=10, 
           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white'))
    
    ax[1].grid(linestyle='--',zorder=0)
    ax[1].scatter(np.log(df['prize_est']),np.log(df['ticket_count']),zorder=5,s=4,color='gray')    
    ax[1].plot(tik_count_reg['plot_x'],tik_count_reg['plot_y'],color='red',zorder=10)
    ax[1].set_xlabel(r'$\ln$(Prize Estimate)')
    ax[1].set_ylabel(r'$\ln$(Ticket Count)')
    ax[1].text(0.05, 0.95, f"Elasticity: {round(tik_count_reg['beta']['prize_est'],2)}", transform=ax[1].transAxes, fontweight='bold', fontsize=10, 
           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white'))
    
    plt.tight_layout()
    plt.savefig(OUTPUTS / 'chart_item_a.pdf')
    plt.close()

def item_e(data):
    '''
    Performs the analysis requested in item (e), and saves the outputs in the file

    Parameters
    ----------
    data : pandas.DataFrame
        Data Frame with the dataset.

    Returns
    -------
    None.

    '''
    
    df: pd.DataFrame = data.copy()
    
    # Finding correlations
    cor_cols: list = ['prize_est','ticket_return']
    cor_tab : list = df[cor_cols].corr()
    
    cor_tab.to_csv(OUTPUTS / 'cor_tab_item_e.csv')
    
    # Estimating elasticities with ticket_return
    reg            : dict = lin_regression(df=df,y=['ticket_return'],x=['prize_est','virada_dummy'],log_vars=['prize_est'],const=True)
    
    plot_x,linear_y       = plot_reg(df=df,reg_res=reg,x_var='prize_est')
    
    plot_x: pd.Series     = pd.Series(plot_x)
    
    # Bin average
    df['bin'] = pd.qcut(np.log(df['prize_est']),q=10)
    
    mean     : pd.Series    = df.groupby('bin')['ticket_return'].mean()
    count    : pd.Series    = df.groupby('bin')['ticket_return'].count()
    bin_reg  : pd.DataFrame = pd.concat([mean,count],axis=1,keys=['mean','count']).reset_index()
    bins     : list         = [bin_reg['bin'].iloc[0].left] + [interval.right for interval in bin_reg['bin']]
    mappoints: pd.Series    = pd.cut(plot_x, bins=bins, include_lowest=True)
    meanmap  : dict         = dict(zip(bin_reg['bin'], bin_reg['mean']))
    
    bin_y    :   pd.Series  = mappoints.map(meanmap)
    
    del([mean,count,bins,mappoints,meanmap])
    
    bin_y[0] = bin_reg['mean'][0]
    bin_y    = bin_y.ffill()
    
    # Kernel Regression
    kern_mod: KernelReg = KernelReg(endog = df['ticket_return'], exog = np.log(df['prize_est']), var_type='c')
    kernel_y, _         = kern_mod.fit(data_predict=plot_x)
    
    # Plotting prize announcement vs revenue and ticket return
    fig,ax = plt.subplots(figsize=(9,5))
    
    ax.grid(linestyle='--',zorder=0)
    ax.scatter(np.log(df['prize_est']),df['ticket_return'],zorder=5,s=4,color='gray')
    ax.plot(plot_x,linear_y,color='red',zorder=10,label='Linear')
    ax.plot(plot_x,bin_y,color='blue',zorder=10,label='Bin Regression')
    ax.plot(plot_x,kernel_y,color='green',zorder=10,label='Kernel Regression')
    ax.set_xlabel(r'$\ln$(Prize Estimate)')
    ax.set_ylabel('Ticket Return')
    ax.legend(loc=0)
    ax.set_ylim([-0.05,1.5])
    ax.axhline(1,color='black',zorder=10)
    
    plt.tight_layout()
    plt.savefig(OUTPUTS / 'chart_item_e.pdf')
    plt.close()
    

# Running the code

if __name__ == '__main__':
    data = pd.read_csv(DATA_FINAL / 'data.csv',index_col=0)
    data['contest_date'] = pd.to_datetime(data['contest_date'], format = '%Y-%m-%d')
    
    # Item (a)
    item_a(data=data)
    item_e(data=data)
    