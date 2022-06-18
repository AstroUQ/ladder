# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 15:42:57 2022

@author: ryanw
"""

import numpy as np
import matplotlib.pyplot as plt

theta = np.arange(0.2, 3 * np.pi, 0.1)
theta2 = np.arange(0.3, 3 * np.pi, 0.1)
x1 = theta * np.cos(theta)
y1 = theta * np.sin(theta)
x2 = theta2 * np.cos(theta2 + np.pi/3)
y2 = theta2 * np.sin(theta2 + np.pi/3)

for [x, y] in [[x1, y1], [x2, y2]]:
    plt.plot(x, y)
    plt.plot(-x, -y)

#BARRED SPIRAL TEST:
# theta = np.arange(np.pi / 2, 1 * np.pi, 0.1)
# x = theta**(1/2) * np.cos(theta)
# y = theta**(1/2) * np.sin(theta)

# w = theta * np.cos(theta + 0.05 * np.pi)
# v = theta * np.sin(theta + 0.05 * np.pi)


# plt.plot(x, y)
# plt.plot(-x, -y)
# plt.plot([0, 0], [1.26, -1.26], 'k')
# plt.plot(w, v)
# plt.xlim(-1, 1); plt.ylim(-1, 1)