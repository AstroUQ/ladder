# -*- coding: utf-8 -*-
"""
Created on Thu Jun  9 21:50:18 2022

@author: ryanw
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

nstars = 1000
# lumin = np.geomspace(0.0001, 2000, nstars)
mainseqtemps = np.geomspace(2800, 30000, nstars)

# a = 1.22887 * 10**-2; b = 4 * 10**-1; c = -0.0375629
# lumin = (b * np.exp(a * mainseqtemps) + c) * (1 + 0.2 * np.sin(mainseqtemps / 5000) + np.random.normal(0, 0.08, nstars))

a = 7.08863; b = -65.4755
mainseqlumin = (a * mainseqtemps + b) * (1 + 0.2 * np.sin(mainseqtemps / 5000) + np.random.normal(0, 0.08, nstars))

# a = 1.4 * 10**-12; b = 30; c = 5; d = -0.652707
# logval = np.log(mainseqtemps) / np.log(b)
# mainseqlumin = (a * 10**(c * logval) + d) * (1 + 0.2 * np.sin(mainseqtemps / 1600) + np.random.normal(0, 0.2, nstars))

colourdata = pd.read_csv("blackbodycolours.txt", delimiter=' ')
mainseqrgb = []
for temperature in mainseqtemps:
    temp = round(temperature / 100) * 100
    mainseqrgb.append(colourdata.loc[colourdata['Temperature'] == temp].iloc[0, 9:12])

mainseqrgb = np.array(mainseqrgb) / 255
fig, ax = plt.subplots()
ax.scatter(mainseqtemps, mainseqlumin, c=mainseqrgb)
ax.set_yscale('log')
ax.set_xscale('log')
ax.invert_xaxis()
ax.set_facecolor('k')
ax.set_xlabel('Temperature (K)'); ax.set_ylabel(r"Solar Luminosities ($L_\odot$)")