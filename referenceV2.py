# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 09:34:52 2017

@author: PapadakisI
"""

"""
Functions with Many Applications
"""

#%% Graph Out List

from array import array

def adjacencyList(E):
    '''
    E: Edge List (Assume Edges Unique)
    
    Returns Out-List of Nodes in Directed Graph
        as dict of sets
    (Economizes Memory / Destroys List E)
    '''
    out = dict()
    while E:
        i,j = E.pop()
        out[i] = out.get(i, array('L')) + array('L',[j])
        out[j] = out.get(j, array('L')) + array('L',[i])
    return out

#%% Breadth First Search (BFS)

def BFS(graph,root):
    """
    Returns BFS Distance from Root Node and 
        Parent Nodes Tracing Path to Each Node from Root
    If Node is unreachable from Root, distance returned in 'Inf'
    """
    
    nodes, edges = graph
    
    assert root in nodes
    adjList = adjacencyList(edges)
    
    distance = {root:0}
    parent = {root:None}
    queue = [root]
    
    while queue:
        current = queue.pop()
        
        for node in adjList.get(current,array('L')):
            if node not in distance:
                queue = [node] + queue
                distance[node] = distance[current] + 1
                parent[node] = current
    
    for node in nodes:
        if node not in distance:
            distance[node] = 'Inf'
            
    return distance, parent

graph = [set(range(11)), 
         [[2, 5], [5, 7], [5, 1], [1, 9], [1, 0], [7, 6], [6, 3], [3, 8], [8, 4]]
         ]
         
print(BFS(graph,0))


#%% Graph Diameter

from array import array 

def graphDiameter(graph):
    
    nodesCount, edges = graph
    
    start = edges[0][0]   
    adjList = adjacencyList(edges)
    
    def distTree(start):
        distance = {start:0}
        stack = array('L',[start])
        
        while stack:
            current = stack.pop()
            for node in adjList.get(current,[]):
                if node not in distance:
                    stack = array('L',[node]) + stack
                    distance[node] = distance[current] + 1
        
        return distance
    
    d0 = distTree(start)
    
    dMax = max(zip(d0.values(),d0.keys()))
    
    d1 = distTree(dMax[1])
    
    result = max([d1.get(node,nodesCount),node] for node in range(nodesCount))[0]
    
    return 'Inf' if result == nodesCount else result


graph = [11, 
         [[2, 5], [5, 7], [5, 1], [1, 9], [1, 0], [7, 6], [6, 3], [3, 8], [8, 4]]
         ]

print(graphDiameter(graph))

graph = [10, 
         [[2, 5], [5, 7], [5, 1], [1, 9], [1, 0], [7, 6], [6, 3], [3, 8], [8, 4]]
         ]

print(graphDiameter(graph))

import random
import datetime
random.seed(2017)
print('Setup')
start = datetime.datetime.now()
n = 10**6
nRep = 2
edges = []
for j in range(n):
    curr = set()
    for _ in range(nRep):
        k = random.randint(0,n-1)
        if k not in curr:
            edges += [[j,k]]
            curr |= {k}
print(datetime.datetime.now()-start)

print('Calculation')
start = datetime.datetime.now()
print(graphDiameter([n,edges]))
print(datetime.datetime.now()-start)

'''
Setup
0:00:06.175428
Calculation
15
0:11:32.559784
'''

#%% Shortest Path (Dijkstra)

'''
A min-priority queue is an abstract data type that provides 3 basic operations: 
    add_with_priority(), decrease_priority() and extract_min().
'''

from bisect import bisect, insort
from random import shuffle

class priorityQueue:
    '''
    Assumes queue elements are unique
    Note: uniqueness should not be affected by add,update operations
    '''
    
    def __init__(self,queue=[0],priority=[0]):
        assert len(queue) == len(priority)
        self.queue = sorted(zip([-p for p in priority],queue))

    def __repr__(self):
        p,q = zip(*self.queue)
        return 'priorityQueue({},{})'.format(list(p),list(q))
        
    def extract_min(self):
        return self.queue.pop()
        
    def add_with_priority(self,member,priority,lowIndex=0):
        insort(self.queue,(-priority,member),lowIndex)
        
    def decrease_priority(self,member,oldPriority,newPriority):
        pos_right = bisect(self.queue,(-oldPriority,member))
        del self.queue[pos_right-1]
        self.add_with_priority(member,newPriority,pos_right-1)
 
        
# Testing

foo = list(range(10))
bar = [4,4,4,2,2,2,1,3,5,6]       
pq = priorityQueue(foo,bar)
print(pq)

pq.decrease_priority(0,4,1)
print(pq)

pq.decrease_priority(2,4,4)
print(pq)

pq.decrease_priority(9,6,7)
print(pq)

#%% Test Priority Queue
           
import datetime
foo = list(range(10**6))
bar = foo[:]
shuffle(foo)
shuffle(bar)


start = datetime.datetime.now()
pq = priorityQueue(foo,bar)
print(datetime.datetime.now()-start)

start = datetime.datetime.now()
pq = priorityQueue(foo[:10**5],bar[:10**5])
for i in range(10**5,2*10**5):
    pq.add_with_priority(foo[i],bar[i])
print(datetime.datetime.now()-start)

pq0 = priorityQueue(foo[:10**5],bar[:10**5])

pq = pq0
start = datetime.datetime.now()
for i in range(10**5):
    pq.decrease_priority(foo[i],bar[i],bar[i+1000])
print(datetime.datetime.now()-start)

#%% SHORTEST PATH TREE

def Dijkstra(graph,distance,source):
    '''
    Generate Shortest Path Tree
    
    Inputs:
        undirected graph as [node set, edge list]
        distance as list of size len(edges) (distance values > 0)
        source as int
        
    Output:
        Shortest Path Distance to Each Node ('Inf' if unreachable)
        Parent Node for Each Node        
    '''
    
    NODES,EDGES = 0,1
    
    #assert source in graph[NODES]
    
    start = datetime.datetime.now()
    step = datetime.datetime.now() 
    edgeLength = dict()
    for k,e in enumerate(graph[EDGES]):
        i,j = e
        edgeLength[(i,j)] = edgeLength[(j,i)] = distance[k]    
    print('Edge Lengths Within {}'.format(datetime.datetime.now()-step))
    
    step = datetime.datetime.now()   
    adj = adjacencyList(graph[EDGES])
    print('Adjacency List Within {}'.format(datetime.datetime.now()-step))
    
    step = datetime.datetime.now()  
    stack = priorityQueue([source],[0])
    distTree = {source:0}
    parent = {source:None}
    print('Initialization Within {}'.format(datetime.datetime.now()-step))

    step = datetime.datetime.now()   
    while stack.queue:
        distCurrNeg, current = stack.extract_min()
        #print('----> ',[distCurrNeg,current])
        for v in adj[current]:
            dNew = -distCurrNeg + edgeLength[(current,v)]
            #print(v,dNew,dOld)
            if v not in distTree:
                stack.add_with_priority(v,dNew)
                distTree[v] = dNew
                parent[v] = current
            elif dNew < distTree[v]:
                stack.decrease_priority(v,distTree[v],dNew)
                distTree[v] = dNew
                parent[v] = current
                
    print('Main Loop Within {}'.format(datetime.datetime.now()-step))
    print('Total Run Time {}'.format(datetime.datetime.now()-start))
    return distTree, parent

#%% Example Graph

n = 12
edges = [[2, 5],[5, 7],[5, 1],[1, 9],[1, 0],[7, 6],[6, 3],[3, 8],[8, 4],[2,10],[10,11],[2,4],[4,11],[7,11],[4,9]]
graph = [set(range(n)),edges]

print('Adjacency List')
print(adjacencyList(edges[:]))
distance = [(i+j)/2 for i,j in edges]
print('Edge Lengths')
for i,e in enumerate(edges):
    print(e,distance[i])
print('Shortest Path Tree')
d,p = Dijkstra(graph,distance,0)
print(d)
print(p)

#%% Large Size Example Graph

import random
random.seed(2017)
print('Setup')
start = datetime.datetime.now()
n = 10**4
edges = list(set(tuple(random.sample(range(n),2)) for i in range(300*n)))
distance = [(i+j)/2 for i,j in edges]
print(datetime.datetime.now()-start)

print('Calculation')
start = datetime.datetime.now()
d,p = Dijkstra([set(range(n)),edges],distance,0)
print(datetime.datetime.now()-start)

#%% Large Size Example Graph (Sparce)

import random
random.seed(2017)
print('Setup')
start = datetime.datetime.now()
n = 10**6
nRep = 5
edges = []
for j in range(n):
    curr = set()
    for _ in range(nRep):
        k = random.randint(0,n-1)
        if k not in curr:
            edges += [[j,k]]
            curr |= {k}
adj = adjacencyList(edges[:])
print(adj[0],max(len(adj[j]) for j in adj))
print(len(adj))
distance = array('f',[(i+j)/2 for i,j in edges])
print(datetime.datetime.now()-start)

#print('Graph Diameter')
#start = datetime.datetime.now()
#print(graphDiameter([set(range(n)),edges[:]]))
#print(datetime.datetime.now()-start)


print('Calculation')
start = datetime.datetime.now()
d,p = Dijkstra([range(n),edges],distance,0)
print(datetime.datetime.now()-start)
print(len(distance))
print(max(d.values()))
print(len(d))

# Node Farthest Away
farNode_dist, farNode = max((d[k],k) for k in d)

farNode_path = []
paren = farNode
while paren is not 0:
    farNode_path += [paren]
    paren = p[paren]
print(len(farNode_path))
    
