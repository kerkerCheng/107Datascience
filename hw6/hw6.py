import numpy as np
from collections import defaultdict
import os


G = []
Neighbors = defaultdict(list)
n = 0
filename = 'hw6_dataset.txt'
with open(os.path.abspath(filename), 'r+') as f:
    for line in f:
        a = int(line.split()[0])+1
        b = int(line.split()[1])+1
        G.append((a, b))
        if a > n:
            n = a
        elif b > n:
            n = b

deg = np.zeros(n+1, dtype=int)
bin = np.zeros(n, dtype=int)
pos = np.zeros(n+1, dtype=int)
vert = np.zeros(n+1, dtype=int)

for edge in G:
    deg[edge[0]] += 1
    deg[edge[1]] += 1
    Neighbors[edge[0]].append(edge[1])
    Neighbors[edge[1]].append(edge[0])
md = np.max(deg)

for v in range(1, n+1):
    bin[deg[v]] += 1

start = 1
for d in range(md+1):
    num = bin[d]
    bin[d] = start
    start += num

for v in range(1, n+1):
    pos[v] = bin[deg[v]]
    vert[pos[v]] = v
    bin[deg[v]] += 1

for d in range(md, 0, -1):
    bin[d] = bin[d-1]
bin[0] = 1

for i in range(1, n+1):
    v = vert[i]


