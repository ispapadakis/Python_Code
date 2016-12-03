# -*- coding: utf-8 -*-
"""
Created on Thu Sep  1 15:15:28 2016

@author: PapadakisI
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Input from .CSV to Dataframe

GC = pd.read_csv('C:\\Users\\papadakisi\\OneDrive - Dun & Bradstreet\\My Samples\\germancredit.csv')

# Basic Info

GC.info()
print(GC.get_dtype_counts())

GC.head()
GC.sample(10)

# Important Metadata

# Decision Variable = GoodCredit
target = 'GoodCredit'

# Num Vars Excludes Target
vars_num  = GC.columns[GC.dtypes == 'int64'] # Also GC.select_dtypes(include=['int64']).columns
vars_num  = vars_num[vars_num != target]
# Char Vars
vars_char = GC.columns[GC.dtypes == 'object']   

# Overall Stats

print(GC.describe())

print(GC.describe(include=['object']))

for v in vars_char:
    cnt = GC[v].value_counts().sort_values(ascending=False)
    print("\nTotal: Categories ",cnt.size,"Cases ",cnt.sum())
    print(cnt[:6])
    if cnt[6:].sum() > 0:
        print("Other: Categories ",cnt[6:].size,"Cases ",cnt[6:].sum())
    print("\n")

# One-Way Effects

target_rate = GC[target].mean()

from statsmodels.graphics.mosaicplot import mosaic
print("LIFT BY CATEGORIES")
for v in vars_char:
    mosaic(GC, [v,target])
    print(pd.crosstab(GC[target],GC[v]))
    #print(GC.groupby(target)[v].value_counts().unstack())
    lift = (GC.groupby(v)[target].mean() / target_rate) - 1
    print(lift)
    print('LIFT: Max {0:5.1%} / Min {1:5.1%} '.format(max(lift),min(lift)))
    plt.show()
    
print("NUMERIC VARIABLE DISTRIBUTION BY TARGET")
for v in vars_num:
    print(v)
    print(GC.groupby(target)[v].mean())
    GC.groupby(target)[v].hist()
    GC.boxplot(v,by=target)
    plt.show()
    
# Stand-out Interactions

import itertools as it
cors = np.corrcoef(GC[vars_num].transpose())
print('Significant Pairwise Correlations')
res = []
for v in it.combinations(range(vars_num.size),2):
    if abs(cors[v]) > 0.5:
        res.append((vars_num[v[0]],vars_num[v[1]],cors[v]))
print(res)        


var_intx = res[0]
     
m_good = plt.scatter(var_intx[0], var_intx[1], marker = 'o', color = 'b', data = GC[:][GC[target]==1])
m_bad = plt.scatter(var_intx[0], var_intx[1], marker = 'o', color = 'r', data = GC[:][GC[target]==0])
plt.xlabel(var_intx[0])
plt.ylabel(var_intx[1])
plt.legend([m_good,m_bad],['Good','Bad'])
plt.show()

intx_0 = pd.qcut(GC[var_intx[0]],4)
intx_1 = pd.qcut(GC[var_intx[1]],4)
rate = pd.DataFrame(GC[target].groupby([intx_0,intx_1]).mean().unstack())
prev = pd.DataFrame(GC[target].groupby([intx_0,intx_1]).size().unstack()/intx_0.size)

(rate/target_rate).plot(kind='barh')

prev.plot(kind='line')

print(prev)

rate.ix['[4, 12]']
rate['[250, 1365.5]']

rate.columns.categories
rate.index.categories
rate.index.names


# Useful Functions
GC['duration2'].idxmin() # Min Index
GC[target].value_counts() # Table
GC[GC.columns[GC.dtypes == 'int64']].idxmin(axis=0) # Col wise Min Index

