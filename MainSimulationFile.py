# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 09:52:26 2022

@author: ryanw

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import colors
import scipy.optimize as opt             # this is to fit two axes in the HR diagram
import scipy.ndimage                     # this is to smooth out the BH radio lobes
from multiprocessing import Pool
# import colour as col
from BlackHole import BlackHole
from Galaxy import Galaxy
from GalaxyCluster import GalaxyCluster
from Star import Star
from Universe import Universe
from time import time

    
def plot_all_dopplers(galaxies):
    fig, (ax, cbar_ax) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [30,1]})
    for galaxy in galaxies:
        galaxy.plot_doppler(fig, ax, cbar_ax, blackhole=True)
def plot_all_2d(galaxies, spikes=False, radio=False):
    fig, ax = plt.subplots()
    for galaxy in galaxies:
        galaxy.plot_2d(fig, ax, spikes=spikes, radio=radio)
def plot_all_3d(galaxies):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    for galaxy in galaxies:
        galaxy.plot_3d(ax, camera=False)
        
def main():
    # np.random.seed(3080)
    # galaxy = Galaxy('SBb', (0,500,100), 1000, 100, cartesian=True)
    # galaxy = Galaxy('cD', (180, 90, 500))
    # galaxy2 = Galaxy('E4', (104, 131, 500), 1000, 100)
    # galaxy3 = Galaxy('Sc', (110, 128, 1000), 1000, 50)
    # galaxies = [galaxy]
    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    # galaxy.plot_3d(ax, camera=False)
    
    # galaxy.plot_radio3d()
    
    # fig, ax = plt.subplots()
    # galaxy.plot_radio_contour(ax)
    # galaxy.plot_RotCurve(newtapprox=False, observed=True)
    # galaxy.plot_HR(isoradii=True, xunit="both", yunit="BolLumMag")
    # ax.set_xlim(-15, 15); ax.set_ylim(-15, 15); ax.set_zlim(-15, 15)
    # ax.set_xlim(-10, 10); ax.set_ylim(-10, 10); ax.set_zlim(-10, 10)

    # plot_all_dopplers(galaxies)
    # plot_all_2d(galaxies, spikes=True, radio=True)
    # galaxy.plot_HR(isoradii=True)
    # fig, ax = plt.subplots()
    # galaxy.plot_2d(fig, ax, spikes=True, radio=True)
    # galaxy2.plot_2d(fig, ax, spikes=True, radio=True)
    # galaxy3.plot_2d(fig, ax, spikes=True, radio=True)
    # t0 = time()
    # cluster = GalaxyCluster((180, 90, 2000), 8)
    # cluster = GalaxyCluster((180, 90, 20000), 50, complexity="Distant")
    
    
    # plot_all_2d(cluster.galaxies)
    # plot_all_3d(cluster.galaxies)
    
    # t1 = time()
    # total = t1 - t0
    # print("Time taken =", total, "s")
    universe = Universe()
    clusters = universe.clusters
    galaxies = [cluster.galaxies for cluster in clusters]
    flatgalaxies = [galaxy for cluster in galaxies for galaxy in cluster]
    plot_all_2d(flatgalaxies)

    
if __name__ == "__main__":
    main()
