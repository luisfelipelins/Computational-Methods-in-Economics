# -*- coding: utf-8 -*-
"""
Created on Sat Feb 28 14:51:30 2026

@author: lfval
"""

import numpy as np
import requests
import pandas as pd
from io import BytesIO
from datetime import datetime as dt
from config import DATA_RAW, DATA_FINAL

# Defining functions

def download_results_data(url):
    '''
    Downloads and saves raw Mega-Sena results data

    Parameters
    ----------
    url : str
        URL to download the raw dataset.

    Returns
    -------
    df : pandas.DataFrame
        Data Frame containing the raw data on Mega-Sena results.

    '''
    
    headers : dict                     = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}
    response: requests.models.Response = requests.get(url,headers=headers,verify=False)
    df      : pd.DataFrame             = pd.read_excel(BytesIO(response.content))
    
    df.to_csv(DATA_RAW / 'raw_results.csv')
    
    return df

def convert_currency_column(col):
    '''
    Converts a column with BRL string data into numeric values

    Parameters
    ----------
    col : pandas.Series
        Series corresponding to the column with BRL string data.

    Returns
    -------
    new_col : pandas.Series
        Series corresponding to the column with BRL numeric data.

    '''
    
    new_col: pd.Series = pd.to_numeric(col.str.replace('R$','').str.replace('.','').str.replace(',','.'))

    return new_col

def calc_expected_prize(final_df):
    '''
    Calculates the expected prize of a contest according to item (c)

    Parameters
    ----------
    final_df : pandas.DataFrame
        Data Frame with the final dataset containing contest results and ticket price.

    Returns
    -------
    exp_prize : pandas.Series
        Series containing the expected prize by contest.

    '''
    
    cols     : list = [['prize_6_nums','winners_6_nums'],
                       ['prize_5_nums','winners_5_nums'],
                       ['prize_4_nums','winners_4_nums']]
    exp_prize: list = []
    
    for i in cols:
        temp: pd.Series = final_df[i[1]].div(final_df['ticket_count']) * final_df[i[0]]
        
        exp_prize.append(temp)
        
    exp_prize: pd.Series = sum(exp_prize)
    
    return exp_prize
    
def create_raw_ticket_price():
    '''
    Creates and saves the raw_ticket_price dataset. It's based on information
    from news website Poder360 and other sources. Further information on the documentation

    Returns
    -------
    raw_ticket_price : pandas.Series
        Series containing the raw ticket prize readjustment dates.

    '''
    
    raw_ticket_price: dict      = {dt(year=2009,month=6,day=1)  :1.75,
                                   dt(year=2009,month=9,day=6)  :2.00,
                                   dt(year=2014,month=5,day=10) :2.50,
                                   dt(year=2015,month=5,day=24) :3.50,
                                   dt(year=2019,month=11,day=10):4.50,
                                   dt(year=2023,month=5,day=3)  :5.00,
                                   dt(year=2025,month=7,day=9)  :6.00}
    raw_ticket_price: pd.Series = pd.Series(raw_ticket_price).reset_index()
    
    raw_ticket_price.columns    = ['contest_date','price']
    
    raw_ticket_price.to_csv(DATA_RAW / 'raw_ticket_price.csv')
    
    return raw_ticket_price
            
    
def treat_datasets(raw_results,raw_ticket_price):
    '''
    Uses raw data for Mega-Sena results and unit ticket prices to create and save
    a unique, clean dataset containing all the information needed in the analysis

    Parameters
    ----------
    raw_results : pandas.DataFrame
        Data Frame containing the raw data on Mega-Sena results.
    raw_ticket_price : pandas.DataFrame
        Data Frame containing the raw unit ticket price data.

    Returns
    -------
    None.

    '''
    
    ## -- Treating results data -- ##
    
    # Renaming and keeping only the desired columns
    renamer_dict: dict            = {'Concurso'            :'contest',
                                     'Data do Sorteio'     :'contest_date',
                                     'Rateio 6 acertos'    :'prize_6_nums',
                                     'Ganhadores 6 acertos':'winners_6_nums',
                                     'Rateio 5 acertos'    :'prize_5_nums',
                                     'Ganhadores 5 acertos':'winners_5_nums',
                                     'Rateio 4 acertos'    :'prize_4_nums',
                                     'Ganhadores 4 acertos':'winners_4_nums',
                                     'Arrecadação Total'   :'revenue',
                                     'Estimativa prêmio'   :'prize_est'}
    results      : pd.DataFrame   = raw_results.rename(columns=renamer_dict)[list(renamer_dict.values())]
    
    # Adjusting the format of currency data and parsing dates as datetime
    columns: list           = ['prize_6_nums','prize_5_nums','prize_4_nums','revenue','prize_est']
    
    results[columns]        = results[columns].apply(convert_currency_column)
    results['contest_date'] = pd.to_datetime(results['contest_date'],format='%d/%m/%Y')
    
    # Adjusting prize_est for the next contest
    results['prize_est']    = results['prize_est'].shift()
    
    # Selecting contests only after 2009 with positive revenue
    results = results.loc[results['contest_date']>dt(year=2009,month=5,day=31)]
    results = results.loc[results['revenue']>0]
    
    ## -- Treating unit ticket price data -- ##
    
    ticket_price: pd.DataFrame   = raw_ticket_price.copy()
    ticket_price                 = ticket_price.set_index('contest_date')
    
    ticket_price.loc[results['contest_date'].iloc[-1]] = np.nan
    
    # Resampling values, to make a daily series of ticket prices
    ticket_price = ticket_price.sort_index()
    ticket_price = ticket_price.resample('D').ffill().ffill().reset_index()
    
    ## -- Constructing the final dataset -- ##
    
    # Merging results to ticket_price by date
    final_df: pd.DataFrame   = pd.merge(results,ticket_price,on='contest_date')
    
    final_df['ticket_count'] = final_df['revenue']/final_df['price']
    
    # Calculating expected prize
    final_df['exp_prize'] = calc_expected_prize(final_df)
    
    # Calculating return of unit ticket
    final_df['ticket_return'] = final_df['exp_prize'].div(final_df['price'])
    
    # Creating Mega da Virada dummy
    cond1       : pd.Series = (final_df['contest_date'].dt.month==12) & (final_df['contest_date'].dt.day==31)
    cond2       : pd.Series = (final_df['contest_date'].dt.month==1) & (final_df['contest_date'].dt.day==1)
    virada_dummy: pd.Series = (cond1*1) + (cond2*1)
    
    final_df['virada_dummy'] = virada_dummy
    
    ## -- Saving the final dataset -- ##
    final_df.to_csv(DATA_FINAL / 'data.csv')
    
    
# Running the code

if __name__ == '__main__':
    results_url      = 'https://servicebus3.caixa.gov.br/portaldeloterias/api/resultados/download?modalidade=Mega-Sena'
    raw_results      = download_results_data(results_url)
    raw_ticket_price = create_raw_ticket_price()
    
    treat_datasets(raw_results,raw_ticket_price)
