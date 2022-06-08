# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 15:42:57 2022

@author: ryanw
"""

import numpy as np
import matplotlib.pyplot as plt

theta = np.arange(np.pi / 2, 1 * np.pi, 0.1)
x = theta**(1/2) * np.cos(theta)
y = theta**(1/2) * np.sin(theta)

# w = theta * np.cos(theta + 0.05 * np.pi)
# v = theta * np.sin(theta + 0.05 * np.pi)


plt.plot(x, y)
plt.plot(-x, -y)
plt.plot([0, 0], [1.26, -1.26], 'k')
# plt.plot(w, v)
# plt.xlim(-1, 1); plt.ylim(-1, 1)