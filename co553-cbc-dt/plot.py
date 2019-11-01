import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
import numpy as np
from tree import DecisionTree

from main2 import *

clean_dataset = np.loadtxt("./wifi_db/clean_dataset.txt")

tree = create_tree(clean_dataset, 10)

run_learning(tree)


start_node = tree.start_node

verts = []
codes = [Path.MOVETO]

for y in range(len(tree.node_list)): #ith layer
    for x in range(len(tree.node_list[y])):

        verts.append((x, -y))
        codes.append(Path.LINETO)
codes.pop(-1)

print(verts)
print(codes)
'''
verts = [
   (0., 0.),   # P0
   (0.2, 1.),  # P1
   (1., 0.8),  # P2
   (0.8, 0.),  # P3
]

codes = [
    Path.MOVETO,
    Path.CURVE4,
    Path.CURVE4,
    Path.CURVE4,
]
'''

path = Path(verts, codes)

fig, ax = plt.subplots()
patch = patches.PathPatch(path, facecolor='none', lw=2)
ax.add_patch(patch)

xs, ys = zip(*verts)
ax.plot(xs, ys, 'x--', lw=2, color='black', ms=10)

ax.text(-0.05, -0.05, 'P0')
ax.text(0.15, 1.05, 'P1')
ax.text(1.05, 0.85, 'P2')
ax.text(0.85, -0.05, 'P3')

ax.set_xlim(-20, 20)
ax.set_ylim(-20, 20)
plt.show()


