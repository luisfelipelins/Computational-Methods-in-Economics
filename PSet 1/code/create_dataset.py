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

# Defining functions

def download_results_data(url):
    headers  = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}
    response = requests.get(url,headers=headers,verify=False)
    df       = pd.read_excel(BytesIO(response.content))
    df.to_csv('data/raw/raw_results.csv')
    
    return df

def convert_currency_column(col):
    new_col = pd.to_numeric(col.str.replace('R$','').str.replace('.','').str.replace(',','.'))

    return new_col
    
def treat_datasets(raw_results,raw_ticket_price):
    
    ## -- Treating results data -- ##
    
    # Renaming and keeping only the desired columns
    renamer_dict            = {'Concurso'            :'contest',
                               'Data do Sorteio'     :'contest_date',
                               'Rateio 6 acertos'    :'prize_6_nums',
                               'Ganhadores 6 acertos':'winners_6_nums',
                               'Rateio 5 acertos'    :'prize_5_nums',
                               'Ganhadores 5 acertos':'winners_5_nums',
                               'Rateio 4 acertos'    :'prize_4_nums',
                               'Ganhadores 4 acertos':'winners_4_nums',
                               'Arrecadação Total'   :'revenue',
                               'Estimativa prêmio'   :'prize_est'}
    results                 = raw_results.rename(columns=renamer_dict)[list(renamer_dict.values())]
    
    # Adjusting the format of currency data and parsing dates as datetime
    columns                 = ['prize_6_nums','prize_5_nums','prize_4_nums','revenue','prize_est']
    results[columns]        = results[columns].apply(convert_currency_column)
    results['contest_date'] = pd.to_datetime(results['contest_date'],format='%d/%m/%Y')
    
    # Adjusting prize_est for the next contest
    results['prize_est']    = results['prize_est'].shift()
    
    # Selecting contests only after 2009 with positive revenue
    results = results.loc[results['contest_date']>dt(year=2009,month=5,day=31)]
    results = results.loc[results['revenue']>0]
    
    ## -- Treating unit ticket price data -- ##
    
    # Renaming and keeping only the desired columns
    renamer_dict = {'anúncio'                                   :'contest_date',
                    'valor da aposta única da Mega-Sena (em R$)':'price'}
    ticket_price = raw_ticket_price.rename(columns=renamer_dict)[list(renamer_dict.values())]
    
    # Adjusting colum formatting
    ticket_price['price']        = pd.to_numeric(ticket_price['price'].str.replace(',','.'))
    ticket_price['contest_date'] = pd.to_datetime(ticket_price['contest_date'].str.replace('mai','may'), format = '%d.%b.%Y')
    ticket_price                 = ticket_price.set_index('contest_date')
    
    # Manually inputting prices before, check documentation for more info
    ticket_price.loc[dt(year=2009,month=9,day=6)]      = 2
    ticket_price.loc[dt(year=2009,month=6,day=1)]      = 1.75
    ticket_price.loc[results['contest_date'].iloc[-1]] = np.nan
    
    # Resampling values, to make a daily series of ticket prices
    ticket_price = ticket_price.sort_index()
    ticket_price = ticket_price.resample('D').ffill().ffill().reset_index()
    
    ## -- Constructing the final dataset -- ##
    
    # Merging results to ticket_price by date
    final_df                 = pd.merge(results,ticket_price,on='contest_date')
    final_df['ticket_count'] = final_df['revenue']/final_df['price']
    
    ## -- Saving the final dataset -- ##
    final_df.to_csv('data/final/data.csv')
    
    

# Running the dataset creating

if __name__ == '__main__':
    results_url      = 'https://servicebus3.caixa.gov.br/portaldeloterias/api/resultados/download?modalidade=Mega-Sena'
    raw_results      = download_results_data(results_url)
    raw_ticket_price = pd.read_csv('data/raw/raw_ticket_prices.csv')
    
    treat_datasets(raw_results,raw_ticket_price)
