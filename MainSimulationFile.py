# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 09:52:26 2022

@author: ryanw

"""

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import colors
from multiprocessing import Pool
import numpy as np
import pandas as pd
import scipy.ndimage                     # this is to smooth out the BH radio lobes
from time import time
# import colour as col
from BlackHole import BlackHole
from Galaxy import Galaxy
from GalaxyCluster import GalaxyCluster
from Star import Star
from Universe import Universe


    
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

class UniverseSim(object):
    def __init__(self, numclusters):
        self.universe = Universe(450000, numclusters)
        self.galaxies = self.universe.get_all_galaxies()
        self.supernovae = self.universe.supernovae
        self.stars = self.universe.get_all_stars()
        self.blackholes = self.universe.get_blackholes()
        
    def plot_universe(self, spikes=True, radio=False):
        '''
        '''
        fig, ax = plt.subplots()
        stars = np.array(self.stars)
        x, y, z, colours, scales = stars[0], stars[1], stars[2], stars[3], stars[4]
        # scales = scales.astype(float)
        equat, polar, radius = self.cartesian_to_spherical(x, y, z)
        
        for i, blackhole in enumerate(self.blackholes):
            BHequat, BHpolar, distance = self.galaxies[i].spherical
            BHcolour = blackhole.get_BH_colour()
            BHscale = blackhole.get_BH_scale() / (0.05 * distance)
            if spikes == True and BHscale > 2.5: 
                spikesize = BHscale / 2
                ax.errorbar(BHequat, BHpolar, yerr=spikesize, xerr=spikesize, ecolor=BHcolour, fmt='none', elinewidth=0.3, alpha=0.5)
            ax.scatter(BHequat, BHpolar, color=BHcolour, s=BHscale)
        
        # scales = [scales[i] / (0.05 * radius[i]) if self.galaxies[i].complexity != "Distant" else scales[i] / (0.001 * radius[i]) for i in range(len(x))]
        # j = [len(stars) for stars in]
        j = np.zeros(len(self.galaxies)+1)
        for i, galaxy in enumerate(self.galaxies):
            pop = len(galaxy.get_stars()[0])
            j[i+1] = pop
        for i, pop in enumerate(j):
            if i != 0:
                j[i] = sum(j[:i])
        scales = [[scales[i+j[k]] / (0.05 * radius[i+j[k]]) if galaxy.complexity != "Distant" else scales[i+j[k]] / (0.001 * radius[i+j[k]]) for i in range(len(galaxy.get_stars()[0]))] for k, galaxy in enumerate(self.galaxies)]
        scales = [scale for galaxy in scales for scale in galaxy]
        if spikes == True:
            brightequat, brightpolar, brightscale, brightcolour = [], [], [], np.empty((0, 3))
            for i, scale in enumerate(scales):
                if scale > 2.5:
                    brightequat += [equat[i]]
                    brightpolar += [polar[i]]
                    brightscale = brightscale + [scale / 4]
                    brightcolour = np.append(brightcolour, [colours[i]], axis=0)
            ax.errorbar(brightequat, brightpolar, yerr=brightscale, xerr=brightscale, ecolor=brightcolour, fmt='none', elinewidth=0.3)
        scales = [3 if scale > 3 else abs(scale) for scale in scales]
        ax.scatter(equat, polar, s=scales, c=colours)
        ax.set_xlim(0, 360); ax.set_ylim(0, 180)
        ax.set_facecolor('k')
        ax.set_aspect(1)
        fig.tight_layout()
        ax.invert_yaxis()
        ax.set_xlabel("Equatorial Angle (degrees)")
        ax.set_ylabel("Polar Angle (degrees)")
        if radio == True:
            self.plot_radio(ax)
    
    def plot_radio(self, ax, plot=True, scatter=False, data=False):
        ''' Plot the radio contours of the SMBH emission onto a 2D sky plot. 
        Parameters
        ----------
        ax : matplotlib axes object
        plot : bool
            Whether to actually plot the contour
        scatter : bool
            Whether to overlay the raw scatter data for calibration purposes
        data : bool
            Whether to return the area density data for the contours
        Returns (if data=True)
        -------
        equatbins, polarbins : numpy arrays (1xN)
            The equatorial and polar coordinates of the contour density bins. 
        density : numpy array (NxN)
            The number count of scatter particles per equat/polar bin. 
        '''
        # x, y, z, radius = [blackhole.get_BH_radio() for blackhole in self.blackholes]
        
        # phi = self.rotation
        # points = np.array([x, y, z])
        # points = np.dot(self.galaxyrotation(phi[0], 'x'), points) # radio scatter is centered at the origin, 
        # points = np.dot(self.galaxyrotation(phi[1], 'y'), points) # so we need to rotate it in the same way as the galaxy was
        # points = np.dot(self.galaxyrotation(phi[2], 'z'), points)
        # x, y, z = points
        # x, y, z = x + self.cartesian[0], y + self.cartesian[1], z + self.cartesian[2] # and now translate it to where the galaxy is
        # equat, polar, distance = self.cartesian_to_spherical(x, y, z)
            
        # extent = [[min(equat) - 3, max(equat) + 3], [min(polar) - 3, max(polar) + 3]]   # this is so that the edge of the contours aren't cut off
        # density, equatedges, polaredges = np.histogram2d(equat, polar, bins=len(equat)//50, range=extent, density=False)
        # equatbins = equatedges[:-1] + (equatedges[1] - equatedges[0]) / 2   # this fixes the order of the bins, and centers the bins at the midpoint
        # polarbins = polaredges[:-1] + (polaredges[1] - polaredges[0]) / 2

        # density = density.T      # take the transpose of the density matrix
        # density = scipy.ndimage.zoom(density, 2)    # this smooths out the data so that it's less boxy and more curvey
        # equatbins = scipy.ndimage.zoom(equatbins, 2)
        # polarbins = scipy.ndimage.zoom(polarbins, 2)
        # # density = scipy.ndimage.gaussian_filter(density, sigma=1)  # this smooths the area density even moreso (not necessary, but keeping for posterity)
        
        equatbins, polarbins, density = [galaxy.plot_radio_contour(0, plot=False, data=True) for galaxy in self.galaxies]
        
        if plot == True:    # plot the contour
            levels = [2, 3, 4, 5, 6, 10, 15]    # having the contour levels start at 2 removes the noise from the smoothing - important!!
            ax.contour(equatbins, polarbins, density, levels, corner_mask=True)     # plot the radio contours
            ax.set_ylim(0, 180); ax.set_xlim(0, 360)
            ax.invert_yaxis();
        if data == True:
            return equatbins, polarbins, density
            # equat/polar are 1xN matrices, whereas density is a NxN matrix. 
    
    def cartesian_to_spherical(self, x, y, z):
        ''' Converts cartesian coordinates to spherical ones (formulae taken from wikipedia) in units of degrees. 
        Maps polar angle to [0, 180] with 0 at the north pole, 180 at the south pole. 
        Maps azimuthal (equatorial) angle to [0, 360], with equat=0 corresponding to the negative x-axis, equat=270 the positive y-axis, etc
        Azimuthal (equat) angles reference (rotates anti-clockwise):
            equat = 0 or 360 -> -ve x-axis (i.e. y=0)
            equat = 90 -> -ve y-axis (x=0)
            equat = 180 -> +ve x-axis (y=0)
            equat = 270 -> +ve y-axis (x=0)
        Parameters
        ----------
        x, y, z : numpy array
            x, y, and z cartesian coordinates
        
        Returns
        -------
        equat, polar, radius : numpy array
            equatorial and polar angles (in degrees), and radius from origin
        '''
        x = x.astype(float); y = y.astype(float); z = z.astype(float)
        radius = np.sqrt(x**2 + y**2 + z**2)
        equat = np.arctan2(y, x)    #returns equatorial angle in radians, maps to [-pi, pi]
        polar = np.arccos(z / radius)
        polar = np.degrees(polar)
        equat = np.degrees(equat)
        # now need to shift the angles
        if np.size(equat) != 1:
            equat = np.array([360 - abs(val) if val < 0 else val for val in equat])  #this reflects negative angles about equat=180
        else:   #same as above, but for a single element. 
            equat = 360 - abs(equat) if equat < 0 else equat
        return (equat, polar, radius)
    
    def spherical_to_cartesian(self, equat, polar, distance):
        '''
        Parameters
        ----------
        equat, polar, distance : numpy array
            equatorial and polar angles, as well as radial distance from the origin
        
        Returns
        -------
        x, y, z : numpy array
            Cartesian coordinates relative to the origin. 
        '''
        equat, polar = np.radians(equat), np.radians(polar)
        x = distance * np.cos(equat) * np.sin(polar)
        y = distance * np.sin(equat) * np.sin(polar)
        z = distance * np.cos(polar)
        return (x, y, z)
        
        
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
    # universe = Universe(450000, 2)
    # flatgalaxies = universe.get_all_galaxies()
    # t0 = time()
    # plot_all_2d(flatgalaxies, spikes=True)
    # t1 = time(); total = t1 - t0; print("Time taken =", total, "s")
    # print(universe.get_all_stars())
    
    sim = UniverseSim(2)
    t0 = time()
    sim.plot_universe()
    t1 = time(); total = t1 - t0; print("Time taken =", total, "s")
    
if __name__ == "__main__":
    main()
