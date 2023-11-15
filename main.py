# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 21:07:33 2023

@author: Gavin
"""

import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
import seaborn as sns
from arch.unitroot.cointegration import engle_granger
from statsmodels.tsa.stattools import coint
import scipy.optimize as optimize
import warnings
import statsmodels.api as sm

warnings.filterwarnings('ignore', category=RuntimeWarning)

os.chdir(r'D:\NUS MFE\FE5221 - Trading Principles and Fundamentals\Assignment')

dir_path = './Data'

all_files = os.listdir(dir_path)


csv_files = [f for f in all_files if f.endswith('.csv')]


all_data = pd.DataFrame()


for csv_file in csv_files:
    file_path = os.path.join(dir_path, csv_file)
    current_data = pd.read_csv(file_path)
    all_data = pd.concat([all_data, current_data], ignore_index=True)
all_data['Analysis Date'] = pd.to_datetime(all_data['Analysis Date'], format='%d/%m/%Y')

"""
Clustering
"""

def generate_data_gics (date , N = 60 , n_clusters  = 70):
    df = all_data.copy()

    df_predict = df[df['Analysis Date'] ==date]
    df_predict['cluster'], _ = pd.factorize(df_predict['GICS Industry'])

    return df_predict

    
    
def generate_data_kmeans(date , N = 60 , n_clusters  = 70 ):
    
    
    df = all_data.copy()


    min_date = date - pd.Timedelta(days=N)
    

    df_last_60_days = df[(df['Analysis Date'] >= min_date) & (df['Analysis Date'] < date)]

    facs_columns = [col for col in df.columns if "FaCS" in col]
    df_facs = df_last_60_days[facs_columns]
    df_facs = df_facs.dropna()
    df_predict = df[df['Analysis Date'] ==date]


    kmeans = KMeans(n_clusters=n_clusters)  
    kmeans.fit_predict(df_facs)
    df_predict_2 = df_predict[facs_columns]
    df_predict_2 = df_predict_2.dropna()

    df_predict['cluster'] = kmeans.predict(df_predict_2)
    
    return df_predict

def generate_data_hc(date , N = 60 , n_clusters  = 70 ):
    
    
    df = all_data.copy()


    min_date = date - pd.Timedelta(days=N)
    

    df_last_60_days = df[(df['Analysis Date'] >= min_date) & (df['Analysis Date'] < date)]
    
    

    facs_columns = [col for col in df.columns if "FaCS" in col]
    df_facs = df_last_60_days[facs_columns]
    df_facs = df_facs.dropna()
    agglomerative = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward", metric="euclidean")
    df_last_60_days['cluster'] = agglomerative.fit_predict(df_facs)
    df_last = df_last_60_days[df_last_60_days['Analysis Date'] ==df_last_60_days['Analysis Date'].max()]
    
    
    # Getting rid of lookahead bias
    df_predict = df[df['Analysis Date'] ==date].merge(df_last[["Asset ID", "cluster"]] , on = "Asset ID" )

    df_predict = df_predict.dropna(subset = ["cluster"])
    return df_predict

def cointegration_test(dataframe ,date, N = 60):
    df = all_data.copy()


    min_date = date - pd.Timedelta(days=N)

    df_last_60_days = df[(df['Analysis Date'] >= min_date) & (df['Analysis Date'] < date)]
    df_last_60_days = df_last_60_days.merge(dataframe[["Asset ID", "cluster"]] , on = "Asset ID" )
    df_last_60_days["Log Price"] = df_last_60_days["Price"].apply(lambda x : np.log(x))
    df_last_60_days = df_last_60_days.dropna(subset = ["cluster"])
    # Group by cluster
    grouped = df_last_60_days.groupby('cluster')
    
    cointegrated_pairs = []
    
    # Perform pairwise cointegration test within each cluster
    for cluster, group in grouped:
        assets = group['Asset ID'].unique()
        for i in range(len(assets)):
            for j in range(i+1, len(assets)):
                asset1 = group[group['Asset ID'] == assets[i]]['Log Price']
                asset2 = group[group['Asset ID'] == assets[j]]['Log Price']
    

                min_length = min(len(asset1), len(asset2))
                asset1 = asset1.iloc[:min_length]
                asset2 = asset2.iloc[:min_length]
    
                if min_length > 0:
                    score, p_value, _ = coint(asset1, asset2)

                    if p_value < 0.05:  
                        cointegrated_pairs.append((assets[i], assets[j], cluster))

    return cointegrated_pairs


def calculate_spread(cointegrated_pairs : tuple , date , N = 60 ):
    df = all_data.copy()


    min_date = date - pd.Timedelta(days=N)
    


    df_last_60_days = df[(df['Analysis Date'] >= min_date) & (df['Analysis Date'] < date)]
    df_last_60_days = df_last_60_days[(df_last_60_days["Asset ID"] == cointegrated_pairs[0]) | (df_last_60_days["Asset ID"] == cointegrated_pairs[1])]
    df_last_60_days["Log Price"] = df_last_60_days["Price"].apply(lambda x : np.log(x))

    pivot_df = df_last_60_days.pivot(index='Analysis Date', columns='Asset ID', values='Log Price')
    

    pivot_df = pivot_df.dropna()
    

    spread = pivot_df[cointegrated_pairs[0]] - pivot_df[cointegrated_pairs[1]]
    
    return spread
    

def calculate_today_spread(cointegrated_pair, date):
    df = all_data.copy()
    df = df[df['Analysis Date'] == date]
    df["Log Price"] = df["Price"].apply(np.log)
    

    log_price_asset1 = df.loc[df['Asset ID'] == cointegrated_pair[0], 'Log Price'].squeeze()
    log_price_asset2 = df.loc[df['Asset ID'] == cointegrated_pair[1], 'Log Price'].squeeze()
    


    if pd.notna(log_price_asset1) and pd.notna(log_price_asset2):

        spread = log_price_asset1 - log_price_asset2
    else:

        spread = np.nan 
    

    return pd.DataFrame({'Spread': [spread]})
    
    
    

def estimate_ou_parameters_closed_form(data, delta_t=1):
    data = data.values
    S_t = data[:-1]
    delta_S_t = data[1:] - data[:-1]

    X = sm.add_constant(S_t)  
    model = sm.OLS(delta_S_t, X)
    results = model.fit()

    alpha, beta = results.params


    kappa = -beta / delta_t
    mu = alpha / (kappa * delta_t)

    sigma = np.std(results.resid) / np.sqrt(delta_t)

    return kappa, mu, sigma

def determine_entry_exit_points_ou_with_stop(results_df, entry_z=1, exit_z=0.5, stop_loss_z=2):
    trades = []

    for _, row in results_df.iterrows():
        kappa = row['Kappa']
        mu = row['Mu']
        sigma = row['Sigma']
        spread = row['Spread']

        in_position = False
        entry_index = None
        exit_index = None
        stop_triggered = False


        z_score = (spread - mu) / sigma

        # Check for entry signal
        if abs(z_score) > entry_z:

            entry_index = row.name  
            in_position = True

        if in_position:

            stop_loss_level = mu + stop_loss_z * sigma

            if spread <= stop_loss_level or abs(z_score) <= exit_z:
      
                exit_index = row.name
                stop_triggered = spread <= stop_loss_level


        if entry_index is not None and exit_index is not None:
            trades.append((entry_index, exit_index, stop_triggered))

    trades_df = pd.DataFrame(trades, columns=['Entry Index', 'Exit Index', 'Stop Triggered'])

    return trades_df

def determine_entry_exit_points_beta_hedged(results_df, date ,  beta_col_name='Beta', entry_z=1, exit_z=0.5, stop_loss_z=2):
    trades = []

    for _, row in results_df.iterrows():
        asset1 = row['Asset1']
        asset2 = row['Asset2']
        kappa = row['Kappa']
        mu = row['Mu']
        sigma = row['Sigma']
        spread = row['Spread']
        beta_1 = row['Beta1']  
        beta_2 = row['Beta2']  


        in_position = False
        entry_index = None
        exit_index = None
        stop_triggered = False


        z_score = (spread - mu) / sigma



        hedge_ratio = beta_1 / beta_2


        if abs(z_score) > entry_z:

            entry_index = row.name  
            in_position = True
            if z_score > 0:
                buy = asset2
                sell = asset1
                trade = 'Convergence'
            elif z_score<0:
                buy = asset1
                sell = asset2
                trade='Divergence'

                


        stop_loss_level = mu + stop_loss_z * sigma
        exit_z = mu +exit_z * sigma
        

        

        if entry_index is not None :

            trades.append({
                'Date':date , 
                'Entry Index': entry_index,
                'Exit Index': exit_index,
                'Hedge Ratio': hedge_ratio,
                'Spread_z': z_score,
                'Buy': buy,
                'Sell':sell,
                'Trade': trade,
                'Target Exit':exit_z,
                "Stop Level": stop_loss_level
                
            })


    trades_df = pd.DataFrame(trades)

    return trades_df

import pandas as pd
from datetime import datetime

def position_tracker(current_positions, date, new_trades, total_aum, max_risk_per_trade=0.01):

    current_positions = current_positions.copy()

    indices_to_drop = []

    for index, position in current_positions.iterrows():
        spread_df = calculate_today_spread((position['Buy'], position['Sell']), date)
        

        if not spread_df.empty and len(spread_df) == 1:
            current_spread_z = spread_df['Spread'].iloc[-1]
        else:
            continue  


        if pd.notna(position['Target Exit']) and pd.notna(position['Stop Level']):
            #Ensure we have scalar values
            target_exit = float(position['Target Exit'])
            stop_level = float(position['Stop Level'])

            if position['Trade'] == 'Divergence':
                if current_spread_z >= target_exit or current_spread_z <= stop_level:
                    indices_to_drop.append(index)
            elif position['Trade'] == 'Convergence':
                if current_spread_z <= target_exit or current_spread_z >= stop_level:
                    indices_to_drop.append(index)
        else:
            print(f"Invalid target exit or stop level for index {index}.")


    current_positions.drop(indices_to_drop, inplace=True)

    for _, trade in new_trades.iterrows():
        if pd.to_datetime(trade['Date']).date() == pd.to_datetime(date).date():
            total_position_size = total_aum * max_risk_per_trade

            hedge_ratio = trade['Hedge Ratio']
            position_size_asset1 = total_position_size / (1 + hedge_ratio)
            position_size_asset2 = position_size_asset1 * hedge_ratio


            trade = trade.copy()
            trade['Position Size Asset1'] = position_size_asset1
            trade['Position Size Asset2'] = position_size_asset2

            current_positions = current_positions.append(trade, ignore_index=True)
    
    return current_positions



if __name__ == "__main__":
    df = all_data.copy()


    columns = [
        'Date', 'Entry Index', 'Exit Index', 'Hedge Ratio', 'Spread_z',
        'Buy', 'Sell', 'Trade', 'Target Exit', 'Stop Level', 'Position Size'
    ]
    current_positions = pd.DataFrame(columns=columns)

    # AUM
    total_aum = 1e6  # 1 million

    # Define the start date for the backtest and the end date
    start_date = (df['Analysis Date'].min() + pd.Timedelta(days=60))

    end_date = df['Analysis Date'].max()

    date_list = df['Analysis Date'].unique()
    filtered_dates = [d for d in date_list if d >= start_date]


    historical_positions = {}

    # Loop over each date in the DataFrame
    for single_date in filtered_dates:



        trade_date = df[df['Analysis Date'] == single_date]
        trade_date["Log Price"] = np.log(trade_date["Price"])


        cluster_df = generate_data_kmeans(single_date)

        # perform cointegration test and get pairs
        cointegrated_pairs = cointegration_test(cluster_df, single_date)

        # prepare results df for this date
        results_df = pd.DataFrame(columns=['Asset1', 'Asset2', 'Kappa', 'Mu', 'Sigma', 'Beta1', 'Beta2', 'Spread'])

        for i in cointegrated_pairs:
            spread_values = calculate_spread(i, single_date)
            if not spread_values.empty:
                kappa, mu, sigma = estimate_ou_parameters_closed_form(spread_values)
                new_row = {
                    'Asset1': i[0],
                    'Asset2': i[1],
                    'Kappa': kappa,
                    'Mu': mu,
                    'Sigma': sigma,
                    'Beta1': trade_date.loc[trade_date['Asset ID'] == i[0], 'Beta (Bmk)'].values[0],
                    'Beta2': trade_date.loc[trade_date['Asset ID'] == i[1], 'Beta (Bmk)'].values[0],
                    'Spread': spread_values.iloc[-1]  # Last observed spread
                }
                results_df = results_df.append(new_row, ignore_index=True)

        # generate trade signals for the current date
        trade_signals_df = determine_entry_exit_points_beta_hedged(results_df, single_date)

        # update current positions with today's trade signals
        current_positions = position_tracker(current_positions, single_date, trade_signals_df, total_aum)


        historical_positions[single_date] = current_positions.copy()


    for date, positions_df in historical_positions.items():
        print(f"Positions for {date}:")
        print(positions_df)

    all_historical_positions = pd.concat(historical_positions.values(), ignore_index=True)

    all_historical_positions.to_csv('historical_positions.csv', index=False)

    
