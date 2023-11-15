# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 20:50:24 2023

@author: Gavin
"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

df = pd.read_excel('Summary.xlsx', sheet_name="Returns")


dates = pd.to_datetime(df['Date'])
kmeans_returns = df['KMeans Return (%)']
benchmark_returns = df['Benchmark Return (%)']
hc_returns = df['HC Return (%)']
gics_returns = df['GICs Return (%)']

plt.figure(figsize=(10, 6))
plt.plot(dates, kmeans_returns, label="KMeans Clustering Return (%)")
plt.plot(dates, hc_returns, label="Hierarchical Clustering Return (%)")
plt.plot(dates, gics_returns, label="GICs Clustering Return (%)")
plt.plot(dates, benchmark_returns, label="Benchmark Return (%)")


plt.text(dates.iloc[-1], kmeans_returns.iloc[-1], f"{kmeans_returns.iloc[-1]:.2f}%")
plt.text(dates.iloc[-1], hc_returns.iloc[-1], f"{hc_returns.iloc[-1]:.2f}%")
plt.text(dates.iloc[-1], gics_returns.iloc[-1], f"{gics_returns.iloc[-1]:.2f}%")
plt.text(dates.iloc[-1], benchmark_returns.iloc[-1], f"{benchmark_returns.iloc[-1]:.2f}%")

plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=5))
plt.gcf().autofmt_xdate()
plt.xlabel("Date")
plt.ylabel("Returns (%)")
plt.title("Periodic Returns Over Time")
plt.legend()
plt.grid(True)
plt.show()


kmeans_cum_returns = kmeans_returns.cumsum()
hc_cum_returns = hc_returns.cumsum()
gics_cum_returns = gics_returns.cumsum()
benchmark_cum_returns = benchmark_returns.cumsum()


plt.figure(figsize=(10, 6))
plt.plot(dates, kmeans_cum_returns, label="KMeans Clustering Return (%)")
plt.plot(dates, hc_cum_returns, label="Hierarchical Clustering Return (%)")
plt.plot(dates, gics_cum_returns, label="GICs Clustering Return (%)")
plt.plot(dates, benchmark_cum_returns, label="Benchmark Return (%)")
plt.text(dates.iloc[-1], kmeans_cum_returns.iloc[-1], f"{kmeans_cum_returns.iloc[-1]:.2f}%")
plt.text(dates.iloc[-1], hc_cum_returns.iloc[-1], f"{hc_cum_returns.iloc[-1]:.2f}%")
plt.text(dates.iloc[-1], gics_cum_returns.iloc[-1], f"{gics_cum_returns.iloc[-1]:.2f}%")
plt.text(dates.iloc[-1], benchmark_cum_returns.iloc[-1], f"{benchmark_cum_returns.iloc[-1]:.2f}%")

plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=5))
plt.gcf().autofmt_xdate()
plt.xlabel("Date")
plt.ylabel("Cumulative Returns (%)")
plt.title("Cumulative Financial Returns Over Time")
plt.legend()
plt.grid(True)
plt.show()

bins = 20 

plt.figure(figsize=(10, 6))
plt.hist(kmeans_returns, bins, alpha=0.5, label='KMeans Returns (%)')
plt.hist(hc_returns, bins, alpha=0.5, label='Hierarchical Clustering Returns (%)')
plt.hist(gics_returns, bins, alpha=0.5, label='GICs Returns (%)')
plt.hist(benchmark_returns, bins, alpha=0.5, label='Benchmark Returns (%)')

plt.xlabel('Returns (%)')
plt.ylabel('Frequency')
plt.title('Histogram of Periodic Returns')
plt.legend()
plt.grid(True)
plt.show()