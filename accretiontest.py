# -*- coding: utf-8 -*-
"""
Created on Sat Oct  1 12:28:38 2022

@author: ryanw
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.ndimage
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

def cartesian_to_spherical(x, y, z):
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

def rotation(angle, axis):
    '''Rotate a point in cartesian coordinates about the origin by some angle along the specified axis. 
    The rotation matrices were taken from https://stackoverflow.com/questions/34050929/3d-point-rotation-algorithm
    Parameters
    ----------
    angle : float
        An angle in radians.
    axis : str
        The axis to perform the rotation on. Must be in ['x', 'y', 'z']
    Returns
    -------
    numpy array
        The transformation matrix for the rotation of angle 'angle'. This output must be used as the first argument within "np.dot(a, b)"
        where 'b' is an 3 dimensional array of coordinates.
    '''
    if axis == 'x':
        return np.array([[1, 0, 0], [0, np.cos(angle), -np.sin(angle)], [0, np.sin(angle), np.cos(angle)]])
    elif axis == 'y':
        return np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]])
    else:
        return np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    

def plot_BH_disk(position, mass, pop, rot=None, mode='scatter'):
    # position = np.array(position) * 3.086 * 10**16
    c = 299792458
    G = 6.637 * 10**-11
    stefBoltz = 5.6704 * 10**-8
    solarLumin = 3.828 * 10**26
    eddLumin = 3.2 * 10**4 * mass * solarLumin
    mass *= 1.98 * 10**30
    radius = (2 * G * mass / c**2)   # schwarzschild  radius in m
    posEquat, posPolar, distance = cartesian_to_spherical(position[0], position[1], position[2])
    angularSize = np.arctan(radius / distance) * (40 * np.pi / 2)
    print(angularSize)
    BlackHole = plt.Circle((posEquat, posPolar), radius=angularSize, color='k')
    
    efficiency = 0.05
    massAccret = eddLumin / (efficiency * c**2)
    
    Rinner = 2 * radius
    Router = 30 * radius
    # distance = np.sqrt(position[0]**2 + position[1]**2 + position[2]**2)
    
    theta = np.random.uniform(0, 2*np.pi, pop)
    vert = 0.1
    phi = np.arccos(np.random.uniform(-vert, vert, pop))
    radii = np.sqrt(np.random.uniform(0, 1, pop) * (Router**2 - Rinner**2) + Rinner**2)
    x = radii * (np.cos(theta) * np.sin(phi) * np.random.normal(1, 0.05, pop))
    y = radii * (np.sin(theta) * np.sin(phi) * np.random.normal(1, 0.05, pop))
    z = radii * (np.cos(phi) * np.random.normal(1, 0.1, pop))
    
    jetpop = int(pop / 14)
    jettheta = np.random.uniform(0, 2*np.pi, jetpop)
    jetx = np.random.uniform(radius, 5 * radius, jetpop) * np.random.choice([-1, 1], jetpop) * np.cos(jettheta)
    jety = np.random.uniform(radius, 5 * radius, jetpop) * np.random.choice([-1, 1], jetpop) * np.sin(jettheta)
    jetz = radius * ((jetx / radius)**2 + (jety / radius)**2) * np.random.choice([-1, 1], jetpop)
    x = np.append(x, jetx)
    y = np.append(y, jety)
    z = np.append(z, jetz)
    
    
    
    
    # tDisk = (3 * G * mass * massAccret / (8 * np.pi * stefBoltz * radius**3))**(1/4)
    # temps = tDisk * (radius / radii)**(3/4) * (1 - np.sqrt(radius / radii))**(1/4) / 100
    # print(min(temps), max(temps))
    
    if rot == None:
        RotPhi = np.random.uniform(0, 2*np.pi, 3)
    else:
        RotPhi = rot
        
    points = np.array([x, y, z])
    points = np.dot(rotation(RotPhi[0], 'x'), points)
    points = np.dot(rotation(RotPhi[1], 'y'), points)
    points = np.dot(rotation(RotPhi[2], 'z'), points)
    x, y, z = points
    x = x + position[0]; y = y + position[1]; z = z + position[2]
    
    equat, polar, radii = cartesian_to_spherical(x, y, z)
    coords = np.array([equat, polar, radii])
    # coords = np.array([[equat[i], polar[i], radii[i], temps[i]] for i in range(len(radii))])
    coords = np.array([[equat[i], polar[i], radii[i]] for i in range(len(radii))])
    coords = coords[coords[:, 2].argsort()]
    plotequat = np.ones(len(equat))
    plotpolar = np.ones(len(equat))
    plotradii = np.ones(len(equat))
    # plottemps = np.ones(len(equat))
    for i, val in enumerate(coords):
        plotequat[i] = val[0]
        plotpolar[i] = val[1]
        plotradii[i] = val[2]
        # plottemps[i] = val[3]
    firstindices = [i for i, d in enumerate(plotradii) if d >= distance]
    secondindices = [i for i, d in enumerate(plotradii) if d < distance]
    
    if mode == 'scatter':
        
        
        # colourdata = pd.read_csv("blackbodycolours.txt", delimiter=' ')
        # rgb = np.ndarray((len(equat), 3))
        # for i, temperature in enumerate(plottemps):
        #     temperature = min(40000, temperature)
        #     temperature = max(1000, temperature)
        #     temp = round(temperature / 100) * 100
        #     r, g, b = colourdata.loc[colourdata['Temperature'] == temp].iloc[0, 9:12]
        #     rgb[i] = np.array([r, g, b]) / 255
        
        
        fig, ax = plt.subplots()
        # ax.scatter(equat, polar, s=0.1, c='w')
        # ax.scatter(plotequat[firstindices], plotpolar[firstindices], s=1, c=rgb[firstindices], alpha=0.1)
        # ax.add_patch(BlackHole)
        # ax.scatter(plotequat[secondindices], plotpolar[secondindices], s=1, c=rgb[secondindices], alpha=0.1)
        ax.scatter(plotequat[firstindices], plotpolar[firstindices], s=1, c='w', alpha=0.1)
        ax.add_patch(BlackHole)
        ax.scatter(plotequat[secondindices], plotpolar[secondindices], s=1, c='w', alpha=0.1)
    else:
        N = 256
        vals = np.ones((N, 4))
        midval = 40
        # start with hubble blue
        for i in range(3):
            vals[:, i] = 1
        vals[midval:, 3] = np.linspace(0, 0.85, N - midval); vals[:midval, 3] = 0
        AccMap = ListedColormap(vals)
        LinearSegmentedColormap('AccMap', AccMap)
        
        vals = np.ones((N, 4))
        # start with hubble blue
        for i in range(3):
            vals[:, i] = 0
        vals[:, 3] = np.linspace(0, 1, N)
        DarkMap = ListedColormap(vals)
        LinearSegmentedColormap('DarkMap', DarkMap)
        
        
        
        
        fig, ax = plt.subplots()
        bins = 300
        # extent = [[min(plotequat) - 3, max(plotequat) + 3], [min(polar) - 3, max(polar) + 3]]   # this is so that the edge of the contours aren't cut off
        density, equatedges, polaredges = np.histogram2d(plotequat[firstindices], plotpolar[firstindices], 
                                                         bins=bins)
        equatbins = equatedges[:-1] + (equatedges[1] - equatedges[0]) / 2   # this fixes the order of the bins, and centers the bins at the midpoint
        polarbins = polaredges[:-1] + (polaredges[1] - polaredges[0]) / 2
        
        density = density.T      # take the transpose of the density matrix
        density = scipy.ndimage.zoom(density, 2)    # this smooths out the data so that it's less boxy and more curvey
        equatbins = scipy.ndimage.zoom(equatbins, 2)
        polarbins = scipy.ndimage.zoom(polarbins, 2)
        density = scipy.ndimage.gaussian_filter(density, sigma=5)  # this smooths the area density even moreso (not necessary, but keeping for posterity)
        ax.pcolormesh(equatbins, polarbins, density, cmap=AccMap)     # plot the radio contours
        
        
        
        
        
        ax.add_patch(BlackHole)
        
        
        
        
        angInner = np.arctan(radius / distance) * (40 * np.pi / 2)
        angOuter = 1.8 * np.arctan(radius / distance) * (40 * np.pi / 2)
        darkpop = int(pop/5)
        darktheta = np.random.uniform(0, 2*np.pi, darkpop)
        darkradii = np.linspace(angInner, angOuter, darkpop)
        darkequat = (darkradii * np.cos(darktheta) * np.random.normal(1, 0.05, darkpop)) + posEquat 
        darkpolar = (darkradii * np.sin(darktheta) * np.random.normal(1, 0.05, darkpop)) + posPolar
        
        density, equatedges, polaredges = np.histogram2d(darkequat, darkpolar, 
                                                          bins=bins)
        equatbins = equatedges[:-1] + (equatedges[1] - equatedges[0]) / 2   # this fixes the order of the bins, and centers the bins at the midpoint
        polarbins = polaredges[:-1] + (polaredges[1] - polaredges[0]) / 2
        
        density = density.T      # take the transpose of the density matrix
        density = scipy.ndimage.zoom(density, 2)    # this smooths out the data so that it's less boxy and more curvey
        equatbins = scipy.ndimage.zoom(equatbins, 2)
        polarbins = scipy.ndimage.zoom(polarbins, 2)
        density = scipy.ndimage.gaussian_filter(density, sigma=20)  # this smooths the area density even moreso (not necessary, but keeping for posterity)
        ax.pcolormesh(equatbins, polarbins, density, cmap=DarkMap)     # plot the radio contours
        
        
        
        
        angInner = 2 * np.arctan(radius / distance) * (40 * np.pi / 2)
        angOuter = 30 * np.arctan(radius / distance) * (40 * np.pi / 2)
        lenspop = int(pop/4 * abs(np.sin(RotPhi[1])))
        lenstheta = np.random.uniform(0, 2*np.pi, lenspop)
        lensradii = np.geomspace(angInner, 0.8 * angOuter, lenspop)
        lensequat = (lensradii * np.cos(lenstheta) * np.random.normal(1, 0.05, lenspop)) + posEquat 
        lenspolar = (lensradii * np.sin(lenstheta) * np.random.normal(1, 0.05, lenspop)) + posPolar
        
        density, equatedges, polaredges = np.histogram2d(lensequat, lenspolar, 
                                                          bins=bins)
        equatbins = equatedges[:-1] + (equatedges[1] - equatedges[0]) / 2   # this fixes the order of the bins, and centers the bins at the midpoint
        polarbins = polaredges[:-1] + (polaredges[1] - polaredges[0]) / 2
        
        density = density.T      # take the transpose of the density matrix
        density = scipy.ndimage.zoom(density, 2)    # this smooths out the data so that it's less boxy and more curvey
        equatbins = scipy.ndimage.zoom(equatbins, 2)
        polarbins = scipy.ndimage.zoom(polarbins, 2)
        density = scipy.ndimage.gaussian_filter(density, sigma=8)  # this smooths the area density even moreso (not necessary, but keeping for posterity)
        ax.pcolormesh(equatbins, polarbins, density, cmap=AccMap, vmax=4)     # plot the radio contours
        
        
        
        
        
        
        
        
        
        
        
        
        
        # extent = [[min(plotequat) - 3, max(equat) + 3], [min(polar) - 3, max(polar) + 3]]   # this is so that the edge of the contours aren't cut off
        density, equatedges, polaredges = np.histogram2d(plotequat[secondindices], plotpolar[secondindices], 
                                                         bins=bins)
        equatbins = equatedges[:-1] + (equatedges[1] - equatedges[0]) / 2   # this fixes the order of the bins, and centers the bins at the midpoint
        polarbins = polaredges[:-1] + (polaredges[1] - polaredges[0]) / 2
        
        density = density.T      # take the transpose of the density matrix
        density = scipy.ndimage.zoom(density, 2)    # this smooths out the data so that it's less boxy and more curvey
        equatbins = scipy.ndimage.zoom(equatbins, 2)
        polarbins = scipy.ndimage.zoom(polarbins, 2)
        density = scipy.ndimage.gaussian_filter(density, sigma=5)  # this smooths the area density even moreso (not necessary, but keeping for posterity)
        ax.pcolormesh(equatbins, polarbins, density, cmap=AccMap)     # plot the radio contours
        
        
        
        
        
        
        
        
        
        
    
    ax.set_facecolor('k')
    ax.invert_yaxis()
    ax.set_aspect('equal')
    
    
plot_BH_disk([1e16, 1e16, 1e16], 1e5, 100000, mode="mesh", rot=[np.pi/3, np.pi/3, np.pi/18])
# plot_BH_disk([1e16, 1e16, 1e16], 1e5, 100000)