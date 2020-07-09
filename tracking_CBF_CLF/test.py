import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import colorConverter
from matplotlib.animation import FuncAnimation
import mpl_toolkits.mplot3d as a3

# curve_pos = lambda t: np.array(
#     [-0.5*np.sin(t),
#     0.5*np.cos(t),
#      0.4 * t])

curve_pos = lambda t: np.array(
    [
    np.min([np.max([0.5 * (t - 0.02 * 200), 0]), 0.5 * 0.02 * 100])
        + np.min([np.max([0.5 * (t - 0.02 * 500), 0]), 0.5 * 0.02 * 100])
        + np.min([np.max([0.5 * (t - 0.02 * 800), 0]), 0.5 * 0.02 * 200]),

    np.min([0.5 * t, 0.5 * 0.02 * 200])
     + np.max([np.min([-0.5 * (t - 0.02 * 300), 0]), -0.5 * 0.02 * 200])
     + np.min([np.max([0.5 * (t - 0.02 * 600), 0]), 0.5 * 0.02 * 200]),

     np.min([0.5 * t, 0.5 * 0.02 * 200])
     + np.max([np.min([-0.5 * (t - 0.02 * 300), 0]), -0.5 * 0.02 * 200])
      + np.min([np.max([0.5 * (t - 0.02 * 600), 0]), 0.5 * 0.02 * 200])])

#
      
      
      
curve = np.array(list(map(curve_pos,np.linspace(0,int(1000)*0.02,int(1000)+1))))
      
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_title("target")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.plot(curve[:, 0], curve[:, 1], curve[:, 2], c='r')
plt.show()