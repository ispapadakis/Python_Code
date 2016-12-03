#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 09:00:32 2016

@author: Yanni Papadakis
"""


from pandas_datareader import data
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

aapl = data.DataReader('AAPL', 'yahoo', '1980-01-01')

print(aapl['Close'])

plt.figure(); aapl['Adj Close'].plot(style = 'b-', logy=True); plt.legend()


from pandas.tools.plotting import lag_plot

plt.figure()
lag_plot(aapl['Adj Close'])

from pandas.tools.plotting import autocorrelation_plot

plt.figure()
autocorrelation_plot(np.diff(aapl['Adj Close']))

plt.plot(np.diff(np.log(aapl['Adj Close'])))

plt.grid(b=True, which='both', color='b', linestyle='--')

plt.hist(np.diff(np.log(aapl['Adj Close'])))

aapl.describe()

aapl['Adj Close']['2008-01-01':'2010-01-01'].plot()
plt.grid(which='both')
