# %%
%matplotlib inline

from mpl_toolkits import mplot3d

import numpy as np
import matplotlib.pyplot as plt


import pandas as pd 

# %%

# filename = '../java/MavenProject/dataset/est_sp_100_v1.csv'
filename = './.est_sp_100_v1.csv'

data = pd.read_csv(filename)

# %%

xdata = data['DiscardSafety']
ydata = data['DeadwoodPoint']
zdata = data['Value']


# %%




# %%

fig = plt.figure()

ax = plt.axes(projection="3d")

ax.set_xlim3d(-1,7)

ax.scatter3D(xdata, ydata, zdata, c=zdata, cmap='Greens')
# ax.plot_trisurf(xdata, ydata, zdata,
#                 cmap='viridis', edgecolor='none')

plt.show()