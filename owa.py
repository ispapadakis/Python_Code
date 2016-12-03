#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 19 15:22:45 2016

@author: Yanni Papadakis
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# For .read_csv, always use header=0 when you know row 0 is the header row
df = pd.read_csv('C:\\Users\\ioannis\\Documents\\My Samples\\germancredit.csv')
df.info()

df.head()

df['GoodCredit'].value_counts()

y = df['GoodCredit']
x = df['amount5']

x_order, x_sorted  = zip(*sorted(enumerate(x), key = lambda i: i[1]))

y_in_x_order = [y[i] for i in x_order]

plt.scatter(x_sorted,y_in_x_order)

y_in_x_order_smoothed = pd.ewma(np.array(y_in_x_order),alpha=0.25)
plt.plot(y_in_x_order_smoothed)

plt.plot(x_sorted,y_in_x_order_smoothed,'r-')
plt.scatter(np.log(x_sorted),y_in_x_order_smoothed)

######################################################

import numpy as np
import statsmodels.api as sm

nsample = 50
sig = 0.25
x1 = np.linspace(0, 20, nsample)
X = np.column_stack((x1, np.sin(x1), (x1-5)**2))
X = sm.add_constant(X)
beta = [5., 0.5, 0.5, -0.02]
y_true = np.dot(X, beta)
y = y_true + sig * np.random.normal(size=nsample)

olsmod = sm.OLS(y, X)
olsres = olsmod.fit()
print(olsres.summary())

# In-Sample Prediction
ypred = olsres.predict(X)
print(ypred)

# Out of Sample Prediction
x1n = np.linspace(20.5,25, 10)
Xnew = np.column_stack((x1n, np.sin(x1n), (x1n-5)**2))
Xnew = sm.add_constant(Xnew)
ynewpred =  olsres.predict(Xnew) # predict out of sample
print(ynewpred)

import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.plot(x1, y, 'o', label="Data")
ax.plot(x1, y_true, 'b-', label="True")
ax.plot(np.hstack((x1, x1n)), np.hstack((ypred, ynewpred)), 'r', label="OLS prediction")
ax.legend(loc="best");

'''
Using formulas can make both estimation and prediction a lot easier
'''

from statsmodels.formula.api import ols

data = {"x1" : x1, "y" : y}

res = ols("y ~ x1 + np.sin(x1) + I((x1-5)**2)", data=data).fit()

list(zip(res.params, olsres.params))