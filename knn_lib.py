# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 10:44:16 2016

@author: PapadakisI
"""
from random import randint, seed
from sklearn.neighbors import NearestNeighbors
import numpy as np
   
seed(1997)
X = np.array( [[randint(0,100),randint(0,100)] for i in range(50)])

nbrs = NearestNeighbors(n_neighbors = 5, radius = 3.0, algorithm='ball_tree', p = 1).fit(X)
distances, indices = nbrs.kneighbors(X)
indices                                           
distances

def test(size=50,target=np.array([[4,4]])):
    rmax = int(np.sqrt(size))
    X = np.array( [[randint(0,rmax),randint(0,rmax)] for i in range(size)])
    
    neigh = NearestNeighbors(n_neighbors = 20, radius = 3.0, algorithm='ball_tree', p = 1)
    neigh.fit(X)
    
    return [X[i] for i in neigh.kneighbors(target)[1]]

test()

from timeit import default_timer as timer

times = []

sizes = [10**i for i in range(2,7)]*3

for n in sizes:

    start = timer()
    lst = test(size=n)
    end = timer()

    times.append(end-start)
    
print(times)

import matplotlib.pyplot as plt

plt.plot(sizes,times, 'ro')
#plt.axis([0, 6, 0, 20])
plt.xscale('log')
plt.yscale('log')
plt.title('log graph')
plt.grid(True)

# calc the trendline
z = np.polyfit(np.log(sizes), np.log(times), 1)
p = np.poly1d(z)
plt.plot(sizes,np.exp(p(np.log(sizes))),"r--")
# the line equation:
print("t = %.2e n^%.6f"%(np.exp(z[1]),z[0]))

plt.show()


size = int(1e6)
rmax = int(np.sqrt(size))
X = np.array( [[randint(0,rmax),randint(0,rmax)] for i in range(size)])

#start = timer()
#nbrs = NearestNeighbors(n_neighbors = 10, radius = 3.0, algorithm='ball_tree', p = 1).fit(X)
#distances, indices = nbrs.kneighbors(X)
#print('Completion Time: ',timer()-start)

start = timer()
nbrs = NearestNeighbors(n_neighbors = 10, radius = 5.0, algorithm='kd_tree', p = 1).fit(X)
distances, indices = nbrs.kneighbors(X)
print('KD Tree Fit Completion Time: ',timer()-start)


start = timer()
nmat = nbrs.radius_neighbors(X,radius=5.0)
print('Radius Neighbors Completion Time: ',timer()-start)
ncount = [len(a) for a in nmat[0]]
plt.hist(ncount)
