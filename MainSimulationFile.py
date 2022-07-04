# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 09:52:26 2022

@author: ryanw

"""

import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import os
import pandas as pd
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
    def __init__(self, numclusters, seed=3080):
        '''
        '''
        np.random.seed(seed)
        self.seed = seed
        self.hubble = 1000
        self.universe = Universe(450000, self.hubble, numclusters)
        self.galaxies = self.universe.get_all_galaxies()
        self.supernovae = self.universe.supernovae
        self.starpositions = self.universe.get_all_starpositions()
        self.blackholes = self.universe.get_blackholes()    
    
    def plot_universe(self, spikes=True, radio=False, save=False):
        '''
        '''
        fig, ax = plt.subplots()
        stars = self.starpositions
        x, y, z, colours, scales = stars[0], stars[1], stars[2], stars[3], stars[4]
        equat, polar, radius = self.cartesian_to_spherical(x, y, z)
        
        for i, blackhole in enumerate(self.blackholes):
            BHequat, BHpolar, distance = self.galaxies[i].spherical
            BHcolour = blackhole.get_BH_colour()
            BHscale = blackhole.get_BH_scale() / (0.05 * distance)
            if spikes == True and BHscale > 2.5: 
                spikesize = BHscale / 2
                ax.errorbar(BHequat, BHpolar, yerr=spikesize, xerr=spikesize, ecolor=BHcolour, fmt='none', elinewidth=0.3, alpha=0.5)
            if save == True:
                ax.scatter(BHequat, BHpolar, color=BHcolour, s=BHscale, linewidths=0)
            else:
                ax.scatter(BHequat, BHpolar, color=BHcolour, s=BHscale)
        
        j = np.zeros(len(self.galaxies) + 1)
        for i, galaxy in enumerate(self.galaxies):
            pop = len(galaxy.get_stars()[0])
            j[i+1] = int(pop)
        cumpops = [int(sum(j[:i + 1])) if i != 0 else int(j[i]) for i in range(len(j))]
        scales = [[scales[i+cumpops[k]] / (0.05 * radius[i+cumpops[k]]) if galaxy.complexity != "Distant" 
                   else scales[i+cumpops[k]] / (0.001 * radius[i+cumpops[k]]) for i in range(len(galaxy.get_stars()[0]))] 
                  for k, galaxy in enumerate(self.galaxies)]
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
        if save == True:
            # ax.scatter(equat, polar, s=scales, c=colours, marker='.')
            ax.scatter(equat, polar, s=scales, c=colours, linewidths=0)
        else:
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
        if save == True:
            return fig
    
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
        
        # equatbins, polarbins, density = [galaxy.plot_radio_contour(0, plot=False, data=True) for galaxy in self.galaxies]
        if plot == True:    # plot the contour
            levels = [2, 3, 4, 5, 6, 10, 15]    # having the contour levels start at 2 removes the noise from the smoothing - important!!
            for galaxy in self.galaxies:
                equatbins, polarbins, density = galaxy.plot_radio_contour(0, plot=False, data=True)
                ax.contour(equatbins, polarbins, density, levels, corner_mask=True)     # plot the radio contours
            ax.set_ylim(0, 180); ax.set_xlim(0, 360)
            ax.invert_yaxis();
        if data == True:
            return equatbins, polarbins, density
        
    def plot_doppler(self, log=True, save=False):
        '''
        '''
        fig, (ax, cbar_ax) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [30,1]})
        stars = self.starpositions
        x, y, z, _, scales = stars[0], stars[1], stars[2], stars[3], stars[4]
        equat, polar, radius = self.cartesian_to_spherical(x, y, z)
        
        obsvel = self.universe.radialvelocities
        
        scales = 1 / (0.01 * np.sqrt(radius))
        
        minvel = min(obsvel); maxvel = max(obsvel)
        if maxvel < -minvel:    # this conditional normalises the colourbar such that v=0 is in the middle of the max and min vel
            maxvel = -minvel
        else:
            minvel = -maxvel
    
        cm = plt.cm.get_cmap('bwr')     # blue => white => red colourmap
        if log:
            red = ax.scatter(equat, polar, c=obsvel, cmap=cm , marker='.', s=scales, linewidths=0,
                          norm=colors.SymLogNorm(linthresh=0.03, vmin=minvel, vmax=maxvel))  # note the colourmap for the redshift amount
        else:
            red = ax.scatter(equat, polar, c=obsvel, vmin=minvel, vmax=maxvel, cmap=cm , marker='.', s=scales,
                             linewidths=0)  # note the colourmap for the redshift amount
        
        cbar = fig.colorbar(red, cax=cbar_ax)   # apply the colourbar to the cbar axes.
        cbar.set_label('Radial Velocity (km/s)', rotation=90)

        ax.set_xlim(0, 360); ax.set_ylim(0, 180)
        ax.set_facecolor('k')
        ax.set_aspect(1)    # sets it to be twice as wide as high, so that angular ratios are preserved
        fig.tight_layout()
        ax.invert_yaxis()
        ax.set_xlabel("Equatorial Angle (degrees)")
        ax.set_ylabel("Polar Angle (degrees)")
        
        if save == True:
            return fig
        
            
    def save_data(self, pic=True, stars=True, distantgalax=True, variable=True, supernovae=True, doppler=[True, False]):
        ''' Generates some data, takes other data, and saves it to the system in a new directory within the file directory.
        Parameters
        ----------
        pic : bool
            Whether to generate and save a 2d plot of the universe.
        stars : bool
            Generate and save star data
        distantgalax : bool
            Generate and save distant galaxy data
        variable : bool
            Generate and save variable star data within a subdirectory
        supernovae : bool
            Generate and save supernovae data
        doppler : list of bool
            First bool in list is whether or not to save a doppler graph with log scale. Second bool is whether to save a linear
            scaled one as well.
        '''
        print("Starting data saving..."); t0 = time()
        # first, initialise the directory where all data will be saved
        self.directory = os.path.dirname(os.path.realpath(__file__))    # this is where this .py file is located on the system
        self.datadirectory = self.directory + f"\\Sim Data {self.seed}"
        if os.path.exists(self.datadirectory):  # if this directory exists, we need to append a number to the end of it
            i = 1
            while os.path.exists(self.datadirectory):   # this accounts for multiple copies that may exist
                self.datadirectory = self.directory + f"\\Sim Data {self.seed} ({i})"   # add the number to the end
                i += 1
            os.makedirs(self.datadirectory)     # now create the duplicate directory with the number on the end
        else:
            os.makedirs(self.datadirectory)     # if directory doesn't exist, create it
        
        if pic:
            fig = self.plot_universe(save=True)
            fig.set_size_inches(12, 6, forward=True)
            fig.savefig(self.datadirectory + '\\Universe Image.png', dpi=1500, bbox_inches='tight', pad_inches = 0.01)
            fig.savefig(self.datadirectory + '\\Universe Image.pdf', dpi=1500, bbox_inches='tight', pad_inches = 0.01)
            
        if stars:   # generate and save star data
            #firstly, get star xyz positions and convert them to equatorial/polar
            starpos = self.starpositions    
            x, y, z, _, _ = starpos[0], starpos[1], starpos[2], starpos[3], starpos[4]
            equat, polar, radius = self.cartesian_to_spherical(x, y, z)
            
            # generate parallax according to p = 1/d (pc) formula, with +/- 1 arcsecond of uncertainty
            parallax = (1 / radius) + np.random.uniform(-0.001, 0.001, len(radius))    
            parallax = [angle if angle >= 0.001 else 0 for angle in parallax]
            parallax = np.around(parallax, decimals=3)
            
            # round position values to 4 decimal places which is about ~1/3 of an arcsecond (or rather 0.0001 degrees)
            equat = np.around(equat, decimals=4); polar = np.around(polar, decimals=4);  
            
            # now generate names for each star
            width = int(-(-np.log10(len(equat)) // 1))   # find the "size" of the number of stars, and round up to the closest decimal
            names = [f"S{i:0{width}d}" for i in range(1, len(equat)+1)]   # generate pretty useless names for each of the stars
            
            # now to work with the band luminosity data. First, get the data for each star in the universe
            blueflux = [[star.bandlumin[0] for star in galaxy.stars] for galaxy in self.galaxies]
            greenflux = [[star.bandlumin[1] for star in galaxy.stars] for galaxy in self.galaxies]
            redflux = [[star.bandlumin[2] for star in galaxy.stars] for galaxy in self.galaxies]
            # now, flatten the above arrays and divide them by the distance to the star, squared
            blueflux = np.array([flux for galaxy in blueflux for flux in galaxy]) / (3.086 * 10**16 * radius)**2
            greenflux = np.array([flux for galaxy in greenflux for flux in galaxy]) / (3.086 * 10**16 * radius)**2
            redflux = np.array([flux for galaxy in redflux for flux in galaxy]) / (3.086 * 10**16 * radius)**2
            blueflux = [format(flux, '.3e') for flux in blueflux]   # now round each data point to 3 decimal places
            greenflux = [format(flux, '.3e') for flux in greenflux]; redflux = [format(flux, '.3e') for flux in redflux]
            
            obsvel = self.universe.radialvelocities     # retrieve the radial velocities from the universe object
            
            # now, write all star data to a pandas dataframe
            stardata = {'Name':names, 'Equatorial':equat, 'Polar':polar,        # units of the equat/polar are in degrees
                        'BlueF':blueflux, 'GreenF':greenflux, 'RedF':redflux,   # units of these fluxes are in W/m^2/nm
                        'Parallax':parallax, 'RadialVelocity':obsvel}           # units of parallax are in arcsec, obsvel in km/s
            starfile = pd.DataFrame(stardata)
            
            starfile.to_csv(self.datadirectory + "\\Star Data.txt", index=None, sep=' ')    # and finally save the dataframe to the directory
            
        if supernovae:
            pos, peak = self.supernovae
            equats = [format(abs(equat), '3.2f') for equat in pos[0]]
            polars = [format(abs(polar), '3.2f') for polar in pos[1]]
            peak = [format(flux, '.3e') for flux in peak]
            supernovadata = {"Equatorial":equats, "Polar":polars, "PeakFlux(W)":peak}
            
            supernovafile = pd.DataFrame(supernovadata)
            supernovafile.to_csv(self.datadirectory + "\\Supernova Data.txt", index=None, sep=' ')
            
        if doppler[0]:
            fig = self.plot_doppler(save=True)
            fig.set_size_inches(12, 6, forward=True)
            fig.savefig(self.datadirectory + '\\Doppler Image Log Scale.png', dpi=1500, bbox_inches='tight', pad_inches = 0.01)
            fig.savefig(self.datadirectory + '\\Doppler Image Log Scale.pdf', dpi=1500, bbox_inches='tight', pad_inches = 0.01)
            if doppler[1]:
                fig = self.plot_doppler(log=False, save=True)
                fig.set_size_inches(12, 6, forward=True)
                fig.savefig(self.datadirectory + '\\Doppler Image Linear Scale.png', dpi=1500, bbox_inches='tight', pad_inches = 0.01)
                fig.savefig(self.datadirectory + '\\Doppler Image Linear Scale.pdf', dpi=1500, bbox_inches='tight', pad_inches = 0.01)
                
        t1 = time(); total = t1 - t0; print("Data generated and saved in =", total, "s")
    
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
    
    
    sim = UniverseSim(5)
    sim.save_data()
    
if __name__ == "__main__":
    main()
