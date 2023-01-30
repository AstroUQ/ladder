# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 14:26:39 2023

@author: ryanw
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

fig = plt.figure(figsize=(12, 10), dpi=80)
fig, ax = plt.subplots(figsize=(12, 10))

# axTop = fig.add_subplot(gs[:2, :]) # top row
# axLeftTop = fig.add_subplot(gs[2, :3]) # just above bottom left corner
# axLeft = fig.add_subplot(gs[3:, :3]) # bottom left corner
# axLeftRight = fig.add_subplot(gs[3:, 3:5]) #just to the right of bottom left corner
# axRight = fig.add_subplot(gs[2:, 5:]) # bottom right corner
# axright = axRight.twinx();

x_true = np.array([1, 2, 3, 4, 5])
y_true = np.array([1, 2, 3, 4, 5])
y_unc = 1

y_rand = np.zeros(len(y_true))

frames = 100
fps = 5

grads = np.zeros(frames)
yints = np.zeros(frames)

def animate(i):
    
    ax.clear()
    
    
    ax.errorbar(x_true, y_true, yerr=y_unc, c='k', fmt='.')
    ax.plot(x_true, y_true, 'b-')
    
    
    
    for j, y in enumerate(y_true):
        y_rand[j] = np.random.normal(y_true[j], y_unc)
        
    ax.scatter(x_true, y_rand, c='r', s=25)
    
    
    # now, lets fit a linear trend to this data. We can do this with a polynomial fit of one degree from numpy:
    grads[i], yints[i] = np.polyfit(x_true, y_rand, 1)
    
    for j in range(i):
        ax.plot(x_true, grads[j] * x_true + yints[j], 'r-', alpha=0.1)
        
    meangrad = np.mean(grads[:i])
    SDgrad = np.std(grads[:i])
    meanyint = np.mean(yints[:i])
    SDyint = np.std(yints[:i])
    ax.plot(x_true, meangrad * x_true + meanyint, 'r-')

    # ax.title(f"$y = {meangrad:.2f}x + {meanyint:.2f}$")
    
    
    ax.set_ylim(-1, 7)
    
    
    fig.suptitle(f"Iteration = {i}      $y = ({meangrad:.2f}^+_- {SDgrad:.2f})x + ({meanyint:.2f}^+_- {SDyint:.2f})$")
    
    print(i)
    fig.tight_layout()
    return fig,

ani = animation.FuncAnimation(fig, animate, frames=frames, interval=int(1000/fps))
plt.show()

ani.save(f'Monte Carlo Uncertainty Propagation.gif', writer='pillow')