# %%
%matplotlib inline

from mpl_toolkits import mplot3d

import numpy as np
import matplotlib.pyplot as plt


import pandas as pd 

# %%

# filename = '../java/MavenProject/dataset/est_sp_100_v1.csv'
# filename = './.est_sp_100_v1.csv'
filename = '../java/MavenProject/dataset/hit_sp_20000_v8.csv'

data = pd.read_csv(filename)

# %%

xdata = data['DiscardSafety']
ydata = data['DeadwoodPoint']
zdata = data['Value']

# %%

fig = plt.figure()

ax = plt.axes(projection="3d")

ax.set_xlim3d(-1,7)

ax.scatter3D(xdata, ydata, zdata, c=zdata, cmap='Greens')
# ax.plot_trisurf(xdata, ydata, zdata,
#                 cmap='viridis', edgecolor='none')

plt.show()

# %%

x = data['Turn']
y = data['TurnVal']

colors = [0,0,0]
fig = plt.figure()
plt.scatter(x, y, c=colors, alpha=0.5)
plt.show()

# %%

x = data['Rank']
y = data['CardVal']

colors = [0,0,0]
fig = plt.figure()
plt.scatter(x, y, c=colors, alpha=0.5)
plt.show()

# %%

x = data['nMeldHit']
y = data['nMeldForm']

colors = [0,0,0]
fig = plt.figure()
plt.scatter(x, y, c=colors, alpha=0.5)
plt.show()

# %%

x = data['Turn']
y = data['TotalVal']

colors = [0,0,0]
fig = plt.figure()
plt.scatter(x, y, c=colors, alpha=0.5)
plt.show()

x = data['Rank']
y = data['TotalVal']

colors = [0,0,0]
fig = plt.figure()
plt.scatter(x, y, c=colors, alpha=0.5)
plt.show()

x = data['nMeldHit']
y = data['TotalVal']

colors = [0,0,0]
fig = plt.figure()
plt.scatter(x, y, c=colors, alpha=0.5)
plt.show()

# %%

x = data['Turn']
y = y_pred.reshape((-1,))

colors = [0,0,0]
fig = plt.figure()
plt.scatter(x, y, c=colors, alpha=0.5)
plt.show()

# %%

x.shape

# %%


x = data['Rank'][:-1]
y = y_pred.reshape((-1,))

colors = [0,0,0]
fig = plt.figure()
plt.scatter(x, y, c=colors, alpha=0.5)
plt.show()

# %%

x = data['nMeldHit']
y = y_pred.reshape((-1,))

colors = [0,0,0]
fig = plt.figure()
plt.scatter(x, y, c=colors, alpha=0.5)
plt.show()

# %%

