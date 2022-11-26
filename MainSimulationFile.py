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
from tqdm import tqdm     # this is a progress bar for a for loop
import MiscTools as misc
from BlackHole import BlackHole
from Galaxy import Galaxy
from GalaxyCluster import GalaxyCluster
from Star import Star
from Universe import Universe

def plot_all_dopplers(galaxies):
    ''' Plot the radial velocities of a list of Galaxy objects onto an image. Mainly to be used for troubleshooting.
    '''
    fig, (ax, cbar_ax) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [30,1]})
    for galaxy in galaxies:
        galaxy.plot_doppler(fig, ax, cbar_ax, blackhole=True)
def plot_all_2d(galaxies, spikes=False, radio=False):
    ''' Plot the positions, colours and brightness of a list of Galaxy objects onto an image. Mainly to be used for troubleshooting.
    '''
    fig, ax = plt.subplots()
    for galaxy in galaxies:
        galaxy.plot_2d(fig, ax, spikes=spikes, radio=radio)
def plot_all_3d(galaxies):
    ''' Plot 3D galaxies from a list of Galaxy objects. Mainly to be used for troubleshooting.
    '''
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    for galaxy in galaxies:
        galaxy.plot_3d(ax, camera=False)

class UniverseSim(object):
    def __init__(self, numclusters, hubble=None, seed=None, blackholes=True, darkmatter=True, mode="Normal",
                 homogeneous=False):
        ''' Generates a Universe object and imports important data from it. 
        Parameters
        ----------
        numclusters : int
            The number of GalaxyCluster objects to populate the universe with.
        hubble : float
            The value of the Hubble constant in units of km/s/Mpc.
        seed : int
            A random seed for which to apply to random number generation in the program.
        mode : str
            One of {"Comprehensive", "Normal", "Basic"} which chooses how complicated the data analysis will be. 
            Comprehensive - some galaxies will and won't have darkmatter/blackholes, and have many stars.
            Normal - all galaxies either will or won't have darkmatter/blackholes, and have many stars.
            Basic - all galaxies either will or won't have darkmatter/blackholes, and have few stars.
        '''
        self.seed = seed if seed != None else int(np.random.uniform(0, 9999)) # randomly choose a <=4 digit seed if one isn't given
        np.random.seed(seed)
        self.universe = Universe(450000, numclusters, hubble, blackholes=blackholes, darkmatter=darkmatter, complexity=mode,
                                 homogeneous=homogeneous)
        self.hasblackhole = blackholes; self.hasdarkmatter = darkmatter; self.homogeneous = homogeneous; self.mode = mode
        self.hubble = self.universe.hubble
        self.galaxies, self.distantgalaxies = self.universe.get_all_galaxies()
        self.allgalaxies = self.galaxies + self.distantgalaxies
        self.supernovae = self.universe.supernovae
        self.starpositions = self.universe.get_all_starpositions()
        self.blackholes = self.universe.get_blackholes()  
        self.datadirectory = False
    
    def create_directory(self):
        ''' Creates the folder for the universe data to be stored in. Will only be created prior to trying to save data. 
        '''
        # first, initialise the directory where all data will be saved
        self.directory = os.path.dirname(os.path.realpath(__file__))    # this is where this .py file is located on the system
        subdirectory = f"\\Datasets\\Sim Data (Clusters; {self.universe.clusterpop}, Seed; {self.seed})"
        self.datadirectory = self.directory + subdirectory
        if os.path.exists(self.datadirectory):  # if this directory exists, we need to append a number to the end of it
            i = 1
            while os.path.exists(self.datadirectory):   # this accounts for multiple copies that may exist
                self.datadirectory = self.datadirectory + f" ({i})"   # add the number to the end
                i += 1
            os.makedirs(self.datadirectory)     # now create the duplicate directory with the number on the end
        else:
            os.makedirs(self.datadirectory)     # if directory doesn't exist, create it
    
    def plot_universe(self, spikes=True, radio=False, save=False):
        ''' Plot all of the stars and distant galaxies in the universe onto a rectangular mapping of the inside of the 
        observable universe sphere. X-axis units are in "Equatorial Angle (degrees)", with Y-axis units in "Polar Angle (degrees)."
        Parameters
        ----------
        spikes : bool
            If true, add diffraction spikes to stars according to their apparent brightness
        radio : bool
            If true, plot the black hole radio lobes overlaid onto the universe image.
        save : bool
            If true, returns the matplotlib figure object so that it may be saved further downstream. 
        Returns
        -------
        fig : matplotlib figure object (if "save" parameter is True)
        '''
        fig, ax = plt.subplots()
        stars = self.starpositions
        x, y, z, colours, scales = stars[0], stars[1], stars[2], stars[3], stars[4]
        equat, polar, radius = misc.cartesian_to_spherical(x, y, z)
        colours = colours[:]    # this fixes an issue when rerunning this program since we want to make a copy of the colours array
        
        for i, blackhole in enumerate(self.blackholes):     # first, plot the black holes at their positions
            if blackhole == False:
                continue
            BHequat, BHpolar, distance = self.allgalaxies[i].spherical  # get the coords of each black hole
            BHcolour = blackhole.get_BH_colour()
            BHscale = blackhole.get_BH_scale() / (0.05 * distance)  # determine the scale of the black hole marker from its intrinsic brightness and distance
            if spikes == True and BHscale > 2.5:    # then we want to plot diffraction spikes on the black hole
                spikesize = BHscale / 2
                ax.errorbar(BHequat, BHpolar, yerr=spikesize, xerr=spikesize, ecolor=BHcolour, fmt='none', elinewidth=0.3, alpha=0.5)
            if save == True:    # we want to get rid of the circle outline on the marker if we're saving the image
                ax.scatter(BHequat, BHpolar, color=BHcolour, s=BHscale, linewidths=0)
            else:   # matplotlib can automatically scale markers when zooming, so we don't need to worry about the circle outlines. 
                ax.scatter(BHequat, BHpolar, color=BHcolour, s=BHscale)
        
        # now, we want to plot all of the stars in the universe. Firstly, we need to determine how large they should appear
        j = np.zeros(len(self.galaxies) + 1)    # initialise an array to store galaxy populations
        for i, galaxy in enumerate(self.galaxies):
            pop = len(galaxy.get_stars()[0])
            j[i+1] = int(pop)       # add this galaxy's population to the aforementioned array
        cumpops = [int(sum(j[:i + 1])) if i != 0 else int(j[i]) for i in range(len(j))]     # determine the cumulative population of the galaxies in their order
        # the cumulative population is needed to keep track of which star is being looked at in the next step
        scales = [[scales[i+cumpops[k]] / (0.05 * radius[i+cumpops[k]]) for i in range(len(galaxy.get_stars()[0]))] 
                  for k, galaxy in enumerate(self.galaxies)]    # get the scale of each star, and divide it by some function of its distance to get its apparent brightness
        scales = [scale for galaxy in scales for scale in galaxy]   # the above operation produces nested lists, so this step flattens the list
        if spikes == True:  # now to plot diffraction spikes on the bright stars
            brightequat, brightpolar, brightscale, brightcolour = [], [], [], np.empty((0, 3))
            for i, scale in enumerate(scales):
                if scale > 2.5:     # we want spikes on this star!
                    brightequat += [equat[i]]
                    brightpolar += [polar[i]]
                    brightscale = brightscale + [scale / 4]
                    brightcolour = np.append(brightcolour, [colours[i]], axis=0)
            # now plot makeshift diffraction spikes with no marker (so that the marker can accurately be plotted later)
            ax.errorbar(brightequat, brightpolar, yerr=brightscale, xerr=brightscale, ecolor=brightcolour, fmt='none', elinewidth=0.3)
        scales = [3 if scale > 3 else abs(scale) for scale in scales]   # limit the maximum size of stars so that an abnormally close star doesnt cover the image
        
        # now we obtain the distant galaxy positions and data, and append them to the star arrays (plotting all in one go saves some time)
        DGspherical = np.array([galaxy.spherical for galaxy in self.distantgalaxies])
        DGequat, DGpolar, DGdists = DGspherical[:, 0], DGspherical[:, 1], DGspherical[:, 2]
        equat, polar = np.append(equat, DGequat), np.append(polar, DGpolar)     # append the distant galaxy data to the end of the star data
        
        DGscales = 1 / (0.0001 * DGdists)   # we want to artifically make distant galaxies a bit larger than stars, since they *should* be brighter and bigger
        for scale in DGscales:
            scales.append(scale)
            colours.append([1.0000, 0.8286, 0.7187])    # this is a nice enough colour to show distant galaxies as. Plotting them based on their actual colour would be too expensive
        
        if save == True:    # as with earlier, plot with no circle outline if saving
            ax.scatter(equat, polar, s=scales, c=colours, linewidths=0)
        else:
            ax.scatter(equat, polar, s=scales, c=colours)
        ax.set_xlim(0, 360); ax.set_ylim(0, 180)    # equatorial angle goes from 0->360, polar 0->180
        ax.set_facecolor('k')   # space has a black background, duh
        ax.set_aspect(1)    # makes it so that the figure is twice as wide as it is tall - no stretching!
        fig.tight_layout()
        ax.invert_yaxis()   # polar angle of 0 is at the top, 180 at the bottom
        ax.set_xlabel("Equatorial Angle (degrees)")
        ax.set_ylabel("Polar Angle (degrees)")
        if radio == True:   # plot the radio overlay
            self.plot_radio(ax)
        if save == True:    # close the figure (so it doesnt pop up during the run) and return the figure to save later.
            plt.close()
            return fig
        
    def cubemap(self, x, y, z):
        ''' Transforms cartesian coordinates into (u, v) coordinates on the 6 faces of a cube, with an index corresponding to 
            which cube face the point is projected on to. 
        Parameters
        ----------
        x, y, z : np.array
            Cartesian coordinates of the point(s)
        Returns
        -------
        uc, vc : np.array
            horiz, vertical coords (respectively) in the corresponding cube face
        index : np.array
            corresponding cube face index of each projected point in (x, y, z):
            {0: front, 1: back, 2: top, 3: bottom, 4: left, 5: right}
        '''
        # initialise arrays
        index = np.zeros(x.size); uc = np.zeros(x.size); vc = np.zeros(x.size)
        # rotate the points so that the local galactic center is centered in the 'front' image
        points = np.array([x, y, z])
        points = np.dot(misc.cartesian_rotation(np.pi, 'y'), points)
        points = np.dot(misc.cartesian_rotation(np.pi / 2, 'x'), points)
        x, y, z = points
        
        # now, let's find which cube face each point is projected to. This algorithm was taken from wikipedia, and adapted
        # for python: https://en.wikipedia.org/wiki/Cube_mapping#Memory_addressing
        for i in range(x.size):
            if x.size == 1: # gotta account for arrays of one value
                X = x; Y = y; Z = z
            else:
                X = x[i]; Y = y[i]; Z = z[i]
            # normalise each vector component so that the output coords are between -x and +x (for example)
            absArray = abs(np.array([X, Y, Z]))
            X /= max(absArray); Y /= max(absArray); Z /= max(absArray)
            # now we can find which cube face the point is projected onto:
            if X > 0 and abs(X) >= abs(Y) and abs(X) >= abs(Z): # point is on: POS X -- front
                uc[i] = -Z
                vc[i] = Y
            elif X < 0 and abs(X) >= abs(Y) and abs(X) >= abs(Z): # NEG X -- back
                uc[i] = Z
                vc[i] = Y
                index[i] = 1
            elif Y > 0 and abs(Y) >= abs(X) and abs(Y) >= abs(Z): # POS Y -- top
                uc[i] = X
                vc[i] = -Z
                index[i] = 2
            elif Y < 0 and abs(Y) >= abs(X) and abs(Y) >= abs(Z): # NEG Y -- bottom
                uc[i] = X
                vc[i] = Z
                index[i] = 3
            elif Z > 0 and abs(Z) >= abs(X) and abs(Z) >= abs(Y): # POS Z -- left
                uc[i] = X
                vc[i] = Y
                index[i] = 4
            else: # Z < 0 and abs(Z) >= abs(X) and abs(Z) >= abs(Y)  # NEG Z -- right
                uc[i] = -X
                vc[i] = Y
                index[i] = 5
        return uc, vc, index
    
    def cubemap_plot(self, uc, vc, index, scales, colours, figAxes, spikes=None):
        ''' Plots the points on each cube face for the cube mapping. 
        Parameters
        ----------
        uc, vc : np.array
            horiz, vert coords (respectively) of each point
        index : np.array
            indices of each point corresponding to their projected cube face
        scales, colours : np.array
            size and colour of each point on the image
        figAxes: list of matplotlib objects
            The 6 figure/ax matplotlib objects corresponding to the plot of each cube face. Must be arranged like:
                [[fig1, ax1], [fig2, ax2], ..., [fig6, ax6]]
        spikes : float
            None by default if no diffraction spikes, but input a number to get diffraction spikes on these coords
            Spiked coords MUST be plotted one at a time, otherwise code changes are needed
        '''
        for i in range(6):
            x, y = uc[index == i], vc[index == i] # get all coords of points on this cube face
            x *= 45; y *= 45 # transforms coords from +/- 1 to +/- 45 degrees
            if spikes == None: # don't need to worry about diffraction spikes, just scatter the points
                if uc.size == 1:
                    figAxes[i][1].scatter(x, y, s=scales, color=colours, linewidths=0)
                else:
                    figAxes[i][1].scatter(x, y, s=scales[index == i], color=colours[index == i], linewidths=0)
            else: # we have spikes, so need to use error bar
                figAxes[i][1].errorbar(x, y, yerr=spikes, xerr=spikes, ecolor=colours, fmt='none', elinewidth=0.3)
        
    def save_cubemap(self, spikes=True):
        ''' Plot all of the stars and distant galaxies in the universe onto a cube map of the inside of the 
        observable universe sphere. X-axis units are in "Equatorial Angle (degrees)", with Y-axis units in "Polar Angle (degrees)."
        Parameters
        ----------
        spikes : bool
            If true, add diffraction spikes to stars according to their apparent brightness

        '''
        directions = ['Front', 'Back', 'Top', 'Bottom', 'Left', 'Right']
        figAxes = []
        for i in range(6):
            fig, ax = plt.subplots(figsize=(9,9))
            ax.set_xlim(-45, 45); ax.set_ylim(-45, 45)    # equatorial angle goes from 0->360, polar 0->180
            ax.set_facecolor('k')   # space has a black background, duh
            ax.set_aspect(1)    # makes it so that the figure is twice as wide as it is tall - no stretching!
            fig.tight_layout()
            ax.set_xlabel("X Position (degrees)")
            ax.set_ylabel("Y Position (degrees)")
            ax.grid()
            figAxes.append([fig, ax])
        stars = self.starpositions
        x, y, z, colours, scales = stars[0], stars[1], stars[2], stars[3], stars[4]
        equat, polar, radius = misc.cartesian_to_spherical(x, y, z)
        colours = colours[:]    # this fixes an issue when rerunning this program since we want to make a copy of the colours array
        
        for i, blackhole in enumerate(self.blackholes):     # first, plot the black holes at their positions
            if blackhole == False:
                continue
            BHequat, BHpolar, distance = self.allgalaxies[i].spherical  # get the coords of each black hole
            BHx, BHy, BHz = self.allgalaxies[i].cartesian
            uc, vc, index = self.cubemap(BHx, BHy, BHz)
            BHcolour = blackhole.get_BH_colour()
            BHscale = blackhole.get_BH_scale() / (0.05 * distance)  # determine the scale of the black hole marker from its intrinsic brightness and distance
            if spikes == True and BHscale > 2.5:    # then we want to plot diffraction spikes on the black hole
                spikesize = BHscale / 2
                self.cubemap_plot(uc, vc, index, 0, BHcolour, figAxes, spikes=spikesize)
            self.cubemap_plot(uc, vc, index, BHscale, BHcolour, figAxes)
        
        # now, we want to plot all of the stars in the universe. Firstly, we need to determine how large they should appear
        j = np.zeros(len(self.galaxies) + 1)    # initialise an array to store galaxy populations
        for i, galaxy in enumerate(self.galaxies):
            pop = len(galaxy.get_stars()[0])
            j[i+1] = int(pop)       # add this galaxy's population to the aforementioned array
        cumpops = [int(sum(j[:i + 1])) if i != 0 else int(j[i]) for i in range(len(j))]     # determine the cumulative population of the galaxies in their order
        # the cumulative population is needed to keep track of which star is being looked at in the next step
        scales = [[scales[i+cumpops[k]] / (0.05 * radius[i+cumpops[k]]) for i in range(len(galaxy.get_stars()[0]))] 
                  for k, galaxy in enumerate(self.galaxies)]    # get the scale of each star, and divide it by some function of its distance to get its apparent brightness
        scales = [scale for galaxy in scales for scale in galaxy]   # the above operation produces nested lists, so this step flattens the list
        if spikes == True:  # now to plot diffraction spikes on the bright stars
            brightX, brightY, brightZ, brightscale, brightcolour = [], [], [], [], np.empty((0, 3))
            for i, scale in enumerate(scales):
                if scale > 2.5:     # we want spikes on this star!
                    brightX += [x[i]]
                    brightY += [y[i]]
                    brightZ += [z[i]]
                    brightscale += [scale / 4]
                    brightcolour = np.append(brightcolour, [colours[i]], axis=0)
            # now plot makeshift diffraction spikes with no marker (so that the marker can accurately be plotted later)
            brightX, brightY, brightZ, brightscale = np.array(brightX), np.array(brightY), np.array(brightZ), np.array(brightscale)
            uc, vc, index = self.cubemap(brightX, brightY, brightZ)
            for i in range(len(uc)): # now plot each of the bright stars individually
                self.cubemap_plot(uc[i], vc[i], index[i], 0, brightcolour[i], figAxes, spikes=brightscale[i])
        scales = [3 if scale > 3 else abs(scale) for scale in scales]   # limit the maximum size of stars so that an abnormally close star doesnt cover the image
        
        # now we obtain the distant galaxy positions and data, and append them to the star arrays (plotting all in one go saves some time)
        DGspherical = np.array([galaxy.spherical for galaxy in self.distantgalaxies])
        DGdists = DGspherical[:, 2]
        # equat, polar = np.append(equat, DGequat), np.append(polar, DGpolar)     # append the distant galaxy data to the end of the star data
       
        DGcartesian = np.array([galaxy.cartesian for galaxy in self.distantgalaxies])
        DGx, DGy, DGz = DGcartesian[:, 0], DGcartesian[:, 1], DGcartesian[:, 2]
        x, y, z = np.append(x, DGx), np.append(y, DGy), np.append(z, DGz)
        
        DGscales = 1 / (0.0001 * DGdists)   # we want to artifically make distant galaxies a bit larger than stars, since they *should* be brighter and bigger
        for scale in DGscales:
            scales.append(scale)
            colours.append([1.0000, 0.8286, 0.7187])    # this is a nice enough colour to show distant galaxies as. Plotting them based on their actual colour would be too expensive
        
        # ax.scatter(equat, polar, s=scales, c=colours, linewidths=0)
        scales = np.array(scales)
        colours = np.array(colours)
        uc, vc, index = self.cubemap(x, y, z)
        self.cubemap_plot(uc, vc, index, scales, colours, figAxes)
        
        if not self.datadirectory:
            self.create_directory()
        
        for i in range(6):
            fig, ax = figAxes[i]
            fig.savefig(self.datadirectory + f'\\{directions[i]}.png', dpi=1500)
        
        plt.close()
    
    def plot_radio(self, ax):
        ''' Plot the radio contours of the SMBH emission onto a 2D sky plot. 
        Parameters
        ----------
        ax : matplotlib axes object
        '''
        levels = [2, 3, 4, 5, 6, 10, 15]    # having the contour levels start at 2 removes the noise from the smoothing - important!!
        for galaxy in self.galaxies:
            if galaxy.blackhole == False or galaxy.blackhole.BHradio == False:
                continue
            equatbins, polarbins, density = galaxy.plot_radio_contour(0, plot=False, data=True)
            _, _, dist = galaxy.spherical   # get the distance to the galaxy in question
            lw = 10 / np.sqrt(dist)     # this makes more distant radio lobes proportionally smaller! nice!
            ax.contour(equatbins, polarbins, density, levels, corner_mask=True, linewidths=lw)     # plot the radio contours
        ax.set_ylim(0, 180); ax.set_xlim(0, 360)
        ax.invert_yaxis();
        
    def plot_doppler(self, log=True, save=False):
        ''' Plot the radial velocities of all of the stars and distant galaxies in the universe in terms of a colour scale,
        where the objects are at their positions in the sky. Similar to the "plot_universe()" function. 
        Parameters
        ----------
        log : bool
            If true (by default), plots the radial velocities with a logarithmically scaled colourbar. If False (I don't recommend),
            plots with a linearly scaled colourbar.
        save : bool
            If true, returns the figure so that it may be saved later. 
        Returns
        -------
        figure : matplotlib figure object
            If save==True, closes the figure and returns it so that it may be saved later on. 
        '''
        fig, (ax, cbar_ax) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [30,1]})   # generate a figure and colourbar with width ratio of 30:1
        stars = self.starpositions  # get all of the star positions 
        x, y, z, _, scales = stars[0], stars[1], stars[2], stars[3], stars[4]
        equat, polar, radius = misc.cartesian_to_spherical(x, y, z)
        
        DGspherical = np.array([galaxy.spherical for galaxy in self.distantgalaxies])   # get all of the distant galaxy positions
        DGequat, DGpolar, DGdists = DGspherical[:, 0], DGspherical[:, 1], DGspherical[:, 2]
        # now, append the distant galaxy data to the star data since, for the sake of the program, they're functionally identical
        equat, polar, radius = np.append(equat, DGequat), np.append(polar, DGpolar), np.append(radius, DGdists)
        
        # get the radial velocities of each object
        obsvel = self.universe.radialvelocities
        DGobsvel = self.universe.distantradialvelocities
        obsvel = np.append(obsvel, DGobsvel)
        
        scales = 1 / (0.01 * np.sqrt(radius))   # determine scale much in the same way as in plot_universe() - further away objects are smaller, but this time doesnt depend on intrinsic brightness
        
        minvel = min(obsvel); maxvel = max(obsvel)
        if maxvel < -minvel:    # this conditional normalises the colourbar such that v=0 is in the middle of the max and min vel
            maxvel = -minvel
        else:
            minvel = -maxvel
    
        cm = plt.cm.get_cmap('bwr')     # blue => white => red colourmap (large negative velocities are blue - blueshifted! Nice!)
        if log:     # plot the object positions with a logarithmically scaled colourmap (looks nice, so is the default option)
            red = ax.scatter(equat, polar, c=obsvel, cmap=cm , marker='.', s=scales, linewidths=0,
                          norm=colors.SymLogNorm(linthresh=0.03, vmin=minvel, vmax=maxvel))  # note the colourmap for the redshift amount
        else:   # plots the objects with a linearly scaled colourmap - i dont recommend it, but it's here anyway
            red = ax.scatter(equat, polar, c=obsvel, vmin=minvel, vmax=maxvel, cmap=cm , marker='.', s=scales,
                             linewidths=0)  # note the colourmap for the redshift amount
        
        cbar = fig.colorbar(red, cax=cbar_ax)   # apply the colourbar to the cbar axes.
        cbar.set_label('Radial Velocity (km/s)', rotation=90)

        ax.set_xlim(0, 360); ax.set_ylim(0, 180)
        ax.set_facecolor('k')   # the background of space is black, duh
        ax.set_aspect(1)    # sets it to be twice as wide as high, so that angular ratios are preserved
        fig.tight_layout()
        ax.invert_yaxis()
        ax.set_xlabel("Equatorial Angle (degrees)")
        ax.set_ylabel("Polar Angle (degrees)")
        
        if save == True:
            plt.close()
            return fig
        
            
    def save_data(self, properties=True, pic=True, radio=True, stars=True, variablestars=True, blackbodies=False, 
                  distantgalax=True, supernovae=True, doppler=[False, False], blackhole=True, rotcurves=False):
        ''' Generates some data, takes other data, and saves it to the system in a new directory within the file directory.
        .pdf images are commented out currently - uncomment them at your own risk! (.pdfs can be in excess of 90mb each!)
        Parameters
        ----------
        properties : bool
            Whether to save the properties of the universe (e.g. variable star parameters, galaxy size, etc)
        pic : bool
            Whether to generate and save a 2d plot of the universe.
        radio : bool
            If true, plot another universe image with radio lobes overlaid.
        stars : bool
            Generate and save star data
        distantgalax : bool
            Generate and save distant galaxy data
        variablestars : bool
            Save variable star data within a subdirectory
        supernovae : bool
            Generate and save supernovae data
        doppler : list of bool
            First bool in list is whether or not to save a doppler graph with log scale. Second bool is whether to save a linear
            scaled one as well.
        blackhole : bool
            Whether to save data from black holes in all galaxies
        rotcurves : bool
            Whether to plot and save the galaxy rotation curves of all resolved galaxies. Also plots and saves galaxy cluster
            rotation curves, provided that they're resolved galaxies and the cluster has a population >= 10 galaxies.
        blackbodies : bool
            If true, plots the blackbody curve for stars in the local galaxy (1 curve for each temperature in 500K increments, so
            e.g. 1 curve for a star of temp 4500K, and another for a temp of 5000K, but not two for 4500K etc), and stores them in 
            a subdirectory under the name "{starname}-Temp:{startemp}"
        '''
        print("Starting data saving..."); t0 = time()
        if not self.datadirectory:
            self.create_directory()
        
        if properties:
            proptime1 = time(); print("Writing universe properties...")
            # now to write the universe properties to a file. They're all pretty self-explanatory.
            text = open(self.datadirectory + '\\Universe Details.txt', "w")
            text.write("Universe Parameters: \n")
            text.write("Parameter                   | Value   \n")
            text.write("----------------------------|---------------------------------------\n")
            text.write(f"Simulation Mode:            | {self.mode} \n")
            text.write(f"Has Black Holes?            | {self.hasblackhole} \n")
            text.write(f"Has Dark Matter?            | {self.hasdarkmatter} \n")
            text.write(f"Universe Homogeneous?       | {self.homogeneous} \n")
            text.write(f"Universe Radius             | {self.universe.radius} (pc)\n")
            text.write(f"Hubble Const.               | {round(self.hubble, 3)} (km/s/Mpc)\n")
            text.write(f"Local Galaxy Type           | {self.galaxies[-1].species} \n")
            text.write(f"Dist. to local galax center | {round(self.galaxies[-1].spherical[2], 2)} (pc)\n")
            text.write(f"Radius of local galaxy      | {round(self.galaxies[-1].radius, 2)} (pc)\n")
            if self.galaxies[-1].blackhole != False:
                text.write(f"Local Galax Black Hole Mass | {round(self.galaxies[-1].blackhole.mass, 2)} Solar Masses \n")
            else:
                text.write("Local Galax Black Hole Mass | N/A \n")
            text.write(f"Number of clusters          | {self.universe.clusterpop}\n")
            text.write(f"Number of galaxies          | {len(self.galaxies)} local and {len(self.distantgalaxies)} distant\n")
            text.write("\n\n")
            text.write("Variable Star Properties    | [RoughAvePeriod  LightcurveShape  PeriodLumGradient  PeriodLumYInt]\n")
            text.write("----------------------------|--------------------------------------------------------------------\n")
            for i, variable in enumerate(self.universe.variablestars):  # now to write the properties of the variable stars to file
                if i == 0:
                    text.write(f"Variable Stars?             | {variable}\n")
                else:
                    length = "Short" if i == 1 else "Long"
                    length = "Longest" if i == 3 else length
                    text.write(f"Variable Class:             | {length}\n")
                    text.write(f"                            | {variable}\n")
                    
            text.close()
            hubblediag = self.universe.plot_hubblediagram(save=True)
            hubblediag.savefig(self.datadirectory + '\\Hubble Diagram.png', dpi=600, bbox_inches='tight', pad_inches = 0.01)
            # hubblediag.savefig(self.datadirectory + '\\Hubble Diagram.pdf', dpi=600, bbox_inches='tight', pad_inches = 0.01)
            HR = self.galaxies[-1].plot_HR(isoradii=True, xunit="both", yunit="BolLumMag", variable=True, save=True)
            HR.savefig(self.datadirectory + '\\Local Galaxy HR Diagram.png', dpi=600, bbox_inches='tight', pad_inches = 0.01)
            # HR.savefig(self.datadirectory + '\\Local Galaxy HR Diagram.pdf', dpi=600, bbox_inches='tight', pad_inches = 0.01)
            
            proptime2 = time(); total = proptime2 - proptime1; print("Universe properties saved in", total, "s")
        
        if pic:     # now save a huge pic of the universe. say goodbye to your diskspace
            pictime1 = time(); print("Generating universe picture...")
            fig = self.plot_universe(save=True)
            fig.set_size_inches(18, 9, forward=True)
            fig.savefig(self.datadirectory + '\\Universe Image.png', dpi=1500, bbox_inches='tight', pad_inches = 0.01)
            # fig.savefig(self.datadirectory + '\\Universe Image.pdf', dpi=200, bbox_inches='tight', pad_inches = 0.01)
            pictime2 = time(); total = pictime2 - pictime1; print("Universe picture saved in", total, "s")
            
            if radio and self.hasblackhole:       # plot radio data too
                print("Generating radio overlay...")
                fig = self.plot_universe(radio=True, save=True)
                fig.set_size_inches(18, 9, forward=True)
                fig.savefig(self.datadirectory + '\\Radio Overlay Image.png', dpi=1500, bbox_inches='tight', pad_inches = 0.01)
                # fig.savefig(self.datadirectory + '\\Radio Overlay Image.pdf', dpi=200, bbox_inches='tight', pad_inches = 0.01)
                pictime3 = time(); total = pictime3 - pictime2; print("Radio overlay picture saved in", total, "s")
            
        if stars:   # generate and save star data
            startime1 = time(); print("Generating star data...")
            #firstly, get star xyz positions and convert them to equatorial/polar
            starpos = self.starpositions    
            x, y, z, _, _ = starpos[0], starpos[1], starpos[2], starpos[3], starpos[4]
            equat, polar, radius = misc.cartesian_to_spherical(x, y, z)
            
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
            obsvel = np.around(obsvel, decimals=2)
            
            # now to append data to the star.txt file as to whether a particular star shows a variable light curve
            variabool = np.zeros(len(names))    # get it? variable but in terms of bools
            k = 0
            for galaxy in self.galaxies:
                for star in galaxy.stars:
                    variabool[k] = star.variable
                    k += 1
            
            # now, write all star data to a pandas dataframe
            stardata = {'Name':names, 'Equatorial':equat, 'Polar':polar,        # units of the equat/polar are in degrees
                        'BlueF':blueflux, 'GreenF':greenflux, 'RedF':redflux,   # units of these fluxes are in W/m^2/nm
                        'Parallax':parallax, 'RadialVelocity':obsvel,           # units of parallax are in arcsec, obsvel in km/s
                        'Variable?':variabool}                                  # outputs 1.0 if variable, 0.0 if not      
            starfile = pd.DataFrame(stardata)
            
            starfile.to_csv(self.datadirectory + "\\Star Data.txt", index=None, sep=' ')    # and finally save the dataframe to the directory
            startime2 = time(); total = startime2 - startime1; print("Star Data.txt saved in", total, "s")
            
        if variablestars:
            vartime1 = time(); print("Saving variable data...")
            variabledirectory = self.datadirectory + "\\Variable Star Data"     # save data within a subfolder
            os.makedirs(variabledirectory)
            names = [f"S{i:0{width}d}" for i in range(1, len(equat)+1)]     # names consistent with star names done earlier
            k = 0
            for galaxy in self.galaxies:
                if galaxy.spherical[2] <= 15000:    # we only want to save variable star data if the stars are close-ish
                    for star in galaxy.stars:
                        if star.variable == True:
                            condition1 = galaxy.spherical[2] <= 5000 and star.variabletype[0] == "Long"
                            condition2 = galaxy.spherical[2] <= 7500 and star.variabletype[0] == "Longest"
                            condition3 = star.variabletype[0] in ["Short", "False"]
                            if condition1 or condition2 or condition3:  # if one of the above criteria are met, we want to save the data
                                starname = names[k]
                                if galaxy.rotate == False:      # must be the local galaxy, so we want to save a pic of the lightcurve
                                    fig = star.plot_lightcurve(save=True)
                                    fig.savefig(variabledirectory + f'\\{starname}.png', dpi=400, bbox_inches='tight', pad_inches = 0.01)
                                times, fluxes = star.lightcurve
                                variabledata = {"Time":times, "NormalisedFlux":fluxes}
                                variablefile = pd.DataFrame(variabledata)
                                variablefile.to_csv(variabledirectory + f"\\{starname}.txt", index=None, sep=' ')
                        k +=1
                else:   # we still need to increment the ticker so that later data is accurate
                    k += len(galaxy.stars)
            
            # now to take and plot the period-luminosity data of the local galaxy!
            periods, lumins = [], []
            for star in self.galaxies[-1].stars:
                if star.variable == True:
                    periods.append(star.period)
                    lumins.append(star.luminosity)
            fig, ax = plt.subplots()
            ax.scatter(periods, lumins, s=0.5)
            ax.set_yscale('log'); ax.set_xlim(xmin=0)
            ax.set_xlabel("Period (hours)"); ax.set_ylabel(r"Log Luminosity ($L / L_\odot$)")
            plt.close()
            fig.savefig(self.datadirectory + '\\Period-Luminosity Data.png', dpi=400, bbox_inches='tight', pad_inches = 0.01)
            
            vartime2 = time(); total = vartime2 - vartime1; print("Variable data saved in", total, "s")
        
        if blackbodies:     # plot and save blackbody curves for stars of a given temperature in the local galaxy
            blacktime1 = time(); print("Generating blackbody curves for local galaxy...")
            blackbodydirectory = self.datadirectory + "\\Local Galaxy Blackbody Curves"     # save data within a subfolder
            os.makedirs(blackbodydirectory)
            names = [f"S{i:0{width}d}" for i in range(1, len(equat)+1)]     # names consistent with star names done earlier
            k = 0
            blackbodytemps = np.arange(0, 50)     # we only want two curves for each 1000K temps, so these are the temps/500
            for galaxy in self.galaxies:
                if galaxy.rotate == False:  # must be the local galaxy, so lets plot some blackbody curves!
                    for i, star in enumerate(galaxy.stars):
                        roundtemp = round(star.temperature/500)
                        if roundtemp in blackbodytemps:
                            fig = star.plot_blackbodycurve(markers=True, visible=True, save=True)
                            fig.savefig(blackbodydirectory + f'\\{names[k]}-Temp={round(star.temperature)}.png', dpi=400, bbox_inches='tight', pad_inches = 0.01)
                            blackbodytemps = np.where(blackbodytemps==roundtemp, 0, blackbodytemps)     # remove this temperature from the pool of temps to plot
                        k += 1
                else:   # we still need to increment the ticker so that later data is accurate
                    k += len(galaxy.stars)
            blacktime2 = time(); total = blacktime2 - blacktime1; print("Blackbody curves saved in", total, "s")
        
        if distantgalax:
            distanttime1 = time(); print("Saving distant galaxy data...")
            sphericals = np.array([galaxy.spherical for galaxy in self.distantgalaxies])
            equat, polar, dists = sphericals[:, 0], sphericals[:, 1], sphericals[:, 2]
            distsMeters = dists * 3.086 * 10**16
            
            bandlumin = np.array([galaxy.bandlumin for galaxy in self.distantgalaxies])
            bluelumin, greenlumin, redlumin = bandlumin[:, 0], bandlumin[:, 1], bandlumin[:, 2]
            blueflux, greenflux, redflux = bluelumin / distsMeters**2, greenlumin / distsMeters**2, redlumin / distsMeters**2  
            blueflux = [format(flux, '.3e') for flux in blueflux]   # now round each data point to 3 decimal places
            greenflux = [format(flux, '.3e') for flux in greenflux]; redflux = [format(flux, '.3e') for flux in redflux]
            
            radii = np.array([galaxy.radius for galaxy in self.distantgalaxies])
            sizes = 2 * np.arctan((radii / 2) / dists)      # gets the apparent size of the galaxy (thanks trig!)
            sizes = np.rad2deg(sizes) * 3600    # this gives the size of the galaxy in units of arcseconds
            sizes = np.around(sizes, decimals=4)
            
            DGobsvel = self.universe.distantradialvelocities
            DGobsvel = np.around(DGobsvel, decimals=2)
            
            width = int(-(-np.log10(len(equat)) // 1))
            names = [f"DG{i:0{width}d}" for i in range(1, len(equat)+1)]
            
            equat = [format(coord, '3.4f') for coord in equat]; polar = [format(coord, '3.4f') for coord in polar]
            
            DGdata = {"Name":names, 'Equatorial':equat, 'Polar':polar,
                      'BlueF':blueflux, 'GreenF':greenflux, 'RedF':redflux,
                      'Size':sizes, 'RadialVelocity':DGobsvel}
            DGfile = pd.DataFrame(DGdata)
            DGfile.to_csv(self.datadirectory + "\\Distant Galaxy Data.txt", index=None, sep=' ')
            distanttime2 = time(); total = distanttime2 - distanttime1; print("Distant galaxy data saved in", total, "s")
            
        if supernovae:
            supertime1 = time(); print("Saving supernova data...")
            pos, peak = self.supernovae
            equats = [format(abs(equat), '3.2f') for equat in pos[0]]
            polars = [format(abs(polar), '3.2f') for polar in pos[1]]
            peak = [format(flux, '.3e') for flux in peak]
            names = [f"SNe{i:0{width}d}" for i in range(1, len(equats)+1)]
            supernovadata = {"Name":names, "Equatorial":equats, "Polar":polars, "PeakFlux(W)":peak}
            
            supernovafile = pd.DataFrame(supernovadata)
            supernovafile.to_csv(self.datadirectory + "\\Supernova Data.txt", index=None, sep=' ')
            supertime2 = time(); total = supertime2 - supertime1; print("Supernova data saved in", total, "s")
            
        if doppler[0]:  # save the doppler image with a log scale
            dopplertime1 = time(); print("Saving doppler image...")
            fig = self.plot_doppler(save=True)
            fig.set_size_inches(18, 9, forward=True)
            fig.savefig(self.datadirectory + '\\Doppler Image Log Scale.png', dpi=1500, bbox_inches='tight', pad_inches = 0.01)
            # fig.savefig(self.datadirectory + '\\Doppler Image Log Scale.pdf', dpi=200, bbox_inches='tight', pad_inches = 0.01)
            dopplertime2 = time(); total = dopplertime2 - dopplertime1; print("Doppler image saved in", total, "s")
            if doppler[1]:
                fig = self.plot_doppler(log=False, save=True)
                fig.set_size_inches(18, 9, forward=True)
                fig.savefig(self.datadirectory + '\\Doppler Image Linear Scale.png', dpi=1500, bbox_inches='tight', pad_inches = 0.01)
                # fig.savefig(self.datadirectory + '\\Doppler Image Linear Scale.pdf', dpi=200, bbox_inches='tight', pad_inches = 0.01)
        
        if blackhole:   # save some data about black holes (radio sources)
            bhtime1 = time(); print("Saving black hole data...")
            equats, polars, BHlumins = [], [], []
            for galaxy in self.allgalaxies:
                if galaxy.blackhole != False:
                    equat, polar, dist = galaxy.spherical
                    dist *= 3.086 * 10**16     # get the distance to the BH in meters
                    lumin = (galaxy.blackhole.luminosity * 3.828 * 10**26) / dist**2    # get the lumin in W/m^2
                    if lumin >= 10**-17:        # we set a hard limit on the distance we can detect black holes
                        BHlumins.append(lumin); equats.append(equat); polars.append(polar)
            
            equats = [format(abs(equat), '3.2f') for equat in equats]
            polars = [format(abs(polar), '3.2f') for polar in polars]
            BHlumins = [format(flux, '.3e') for flux in BHlumins]
            
            names = [f"RS{i:0{width}d}" for i in range(1, len(BHlumins)+1)]
            BHdata = {'Name':names, 'Equatorial':equats, 'Polar':polars,
                      'Luminosity':BHlumins}
            BHfile = pd.DataFrame(BHdata)
            BHfile.to_csv(self.datadirectory + "\\Radio Source Data.txt", index=None, sep=' ')
            bhtime2 = time(); total = bhtime2 - bhtime1; print("Black hole data saved in", total, "s")
        
        if rotcurves:   # plot and save the rotation curve of each resolved galaxy
            rottime1 = time(); print("Saving galaxy rotation curves...")
            rotcurvedirectory = self.datadirectory + "\\Galaxy Rotation Curves"
            os.makedirs(rotcurvedirectory)  # make a new folder to hold the rotation curves
            
            for galaxy in self.galaxies:
                equat, polar, _ = galaxy.spherical; equat, polar = round(equat, 2), round(polar, 2)
                bh = "1" if galaxy.blackhole != False else "0"
                dm = "1" if galaxy.darkmatter == True else "0"
                fig = galaxy.plot_RotCurve(newtapprox=True, save=True)
                fig.savefig(rotcurvedirectory + f'\\E{equat}-P{polar} {galaxy.species}, BH{bh}, DM{dm}.png', 
                            dpi=200, bbox_inches='tight', pad_inches = 0.01)
            rottime2 = time(); total = rottime2 - rottime1; print("Galaxy rotation curves saved in", total, "s")
            
            print("Saving cluster rotation curves...")
            clustercurvedirectory = self.datadirectory + "\\Cluster Rotation Curves"
            os.makedirs(clustercurvedirectory)
            for cluster in self.universe.clusters:
                pop = cluster.clusterpop
                if pop >= 10 and cluster.complexity != "Distant":
                    equat, polar, _ = cluster.spherical; equat, polar = round(equat, 2), round(polar, 2)
                    fig = cluster.plot_RotCurve(newtapprox=True, save=True)
                    fig.savefig(clustercurvedirectory + f'\\E{equat}-P{polar}, Pop;{pop}.png', 
                                dpi=400, bbox_inches='tight', pad_inches = 0.01)
            rottime3 = time(); total = rottime3 - rottime2; print("Cluster rotation curves saved in", total, "s")
        t1 = time(); total = t1 - t0; print("All data generated and saved in =", total, "s")
        plt.close()     # need this to close the figure (since the "fig" variable persists)
    
    def savepic(self, directory, radio, proj='AllSky'):
        if proj == 'AllSky':
            pictime1 = time(); print("Generating universe picture...")
            fig = self.plot_universe(save=True)
            fig.set_size_inches(18, 9, forward=True)
            fig.savefig(directory + '\\Universe Image.png', dpi=1500, bbox_inches='tight', pad_inches = 0.01)
            # fig.savefig(self.datadirectory + '\\Universe Image.pdf', dpi=200, bbox_inches='tight', pad_inches = 0.01)
            pictime2 = time(); total = pictime2 - pictime1; print("Universe picture saved in", total, "s")
            
            if radio and self.hasblackhole:       # plot radio data too
                print("Generating radio overlay...")
                fig = self.plot_universe(radio=True, save=True)
                fig.set_size_inches(18, 9, forward=True)
                fig.savefig(directory + '\\Radio Overlay Image.png', dpi=1500, bbox_inches='tight', pad_inches = 0.01)
                # fig.savefig(self.datadirectory + '\\Radio Overlay Image.pdf', dpi=200, bbox_inches='tight', pad_inches = 0.01)
                pictime3 = time(); total = pictime3 - pictime2; print("Radio overlay picture saved in", total, "s")
        
        elif proj == 'Cube':
            pass
        return None
    
        
        
def main():
    ### -- this was used to find galaxy averages -- ###
    # number = 40
    # specieslist = ['S0', 'Sa', 'Sb', 'Sc', 'SBa', 'SBb', 'SBc', 'cD', 'E0', 'E1', 'E2', 'E3', 'E4', 'E5', 'E6', 'E7']
    # for species in specieslist:
    #     galaxies = []
    #     for i in tqdm(range(number)):
    #         galaxies.append(Galaxy(species, (0, 0, 0)))
    #     # galaxies = [Galaxy(species, (0, 0, 0)) for i in range(number)]
    #     masses = [galaxy.galaxymass for galaxy in galaxies]
    #     radii = [galaxy.radius for galaxy in galaxies]
    #     bluef = []; greenf = []; redf = []
    #     for galaxy in galaxies:
    #         for star in galaxy.stars:
    #             bluef.append(star.bandlumin[0])
    #             greenf.append(star.bandlumin[1])
    #             redf.append(star.bandlumin[2])
    #     meanbluef, meangreenf, meanredf = np.mean(bluef), np.mean(greenf), np.mean(redf)
    #     sdbluef, sdgreenf, sdredf = np.std(bluef), np.std(greenf), np.std(redf)
    #     print(f"Galaxy Mass for {number} {species} galaxies: Mean =", np.mean(masses), "with SD =", np.std(masses))
    #     print(f"Galaxy Radius for {number} {species} galaxies: Mean =", np.mean(radii), "with SD =", np.std(radii))
    #     print(f"Galaxy bandlumin for {number} {species} galaxies: Mean =", [meanbluef, meangreenf, meanredf], 
    #           "with SD =", [sdbluef, sdgreenf, sdredf])
    
    ### -- this is the function that you should run! -- ###
    # sim = UniverseSim(1000, mode="Normal")
    # sim.save_data()
    
    sim = UniverseSim(20)
    sim.save_cubemap()
    

    
if __name__ == "__main__":
    main()
