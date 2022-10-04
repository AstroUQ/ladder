# -*- coding: utf-8 -*-
"""
Created on Fri Sep 16 17:21:53 2022

@author: ryanw
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

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


N = 256
valsR = np.ones((N, 4))
valsG = np.ones((N, 4))
valsB = np.ones((N, 4))
midval = 180
# start with hubble blue
valsB[:, 0] = 0
valsB[:midval, 1] = 178 / 256; valsB[midval:, 1] = np.linspace(178 / 256, 43/256, N - midval)
valsB[:midval, 2] = 255 / 256; valsB[midval:, 2] = np.linspace(255 / 256, 29/256, N - midval)
valsB[20:, 3] = np.linspace(0, 0.6, N - 20); valsB[:20, 3] = 0
HubbleBlue = ListedColormap(valsB)
LinearSegmentedColormap('HubbleBlue', valsB)

valsG[:midval, 0] = 200 / 256; valsG[midval:, 0] = np.linspace(200 / 256, 66/256, N - midval)
valsG[:midval, 1] = 255 / 256; valsG[midval:, 1] = np.linspace(255 / 256, 43/256, N - midval)
valsG[:, 2] = 0
valsG[20:, 3] = np.linspace(0, 0.6, N - 20); valsG[:20, 3] = 0
HubbleGreen = ListedColormap(valsG)
LinearSegmentedColormap('HubbleGreen', valsG)

valsR[:midval, 0] = 255 / 256; valsR[midval:, 0] = np.linspace(255 / 256, 66/256, N - midval)
valsR[:, 1] = 0
valsR[:, 2] = 0
valsR[20:, 3] = np.linspace(0, 0.6, N - 20); valsR[:20, 3] = 0
HubbleRed = ListedColormap(valsR)
LinearSegmentedColormap('HubbleRed', valsR)

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

# img = plt.imread("Datasets/Sim Data (Clusters; 1000, Seed; 588)/Universe Image.png")

fig, ax = plt.subplots()
# ax.imshow(img, extent=[0, 360, 0, 180])
radii = [0.15, 0.12, 0.08]
pops = [150000, 120000, 100000]
lowers = [0, 0.1, 0.2]
RotPhi = np.random.uniform(0, 2*np.pi, 3)
for i, colour in enumerate([HubbleRed, HubbleGreen, HubbleBlue]):
    pop = pops[i]
    radius = radii[i] * (np.random.uniform(lowers[i], 1, pop))**(1/3)
    # the following distributes points more or less evenly about a sphere centered at the origin
    theta = np.random.uniform(0, 2*np.pi, pop)
    vert = 0.6 if i < 2 else 1
    phi = np.arccos(np.random.uniform(-vert, vert, pop))
    centerx = radius * (np.cos(theta) * np.sin(phi) * np.random.normal(1, 0.1, pop))
    centery = radius * (np.sin(theta) * np.sin(phi) * np.random.normal(1, 0.1, pop))
    zmult = 1 if i < 2 else 1.5
    centerz = zmult * radius * (np.cos(phi) * np.random.normal(1, 0.3, pop))
    if i == 0:
        outPop = int(0.75 * pop); outRad = 1.5 * radii[i]
        outPhi = np.arccos(np.random.uniform(-1, 1, outPop))
        outTheta = np.random.uniform(0, 2*np.pi, outPop)
        centerx = np.append(centerx, outRad * (np.cos(outTheta) * np.sin(outPhi) * np.random.normal(1, 0.1, outPop)))
        centery = np.append(centery, outRad * (np.sin(outTheta) * np.sin(outPhi) * np.random.normal(1, 0.1, outPop)))
        centerz = np.append(centerz, outRad * (np.cos(outPhi) * np.random.normal(1, 0.3, outPop)))
    
    points = np.array([centerx, centery, centerz])
    points = np.dot(rotation(RotPhi[0], 'x'), points)
    points = np.dot(rotation(RotPhi[1], 'y'), points)
    points = np.dot(rotation(RotPhi[2], 'z'), points)
    
    position = (5, 5, 5)
    x = points[0] + position[0]
    y = points[1] + position[1]
    z = points[2] + position[2]
    
    equat, polar, distance = cartesian_to_spherical(x, y, z)
        
    extent = [[min(equat) - 3, max(equat) + 3], [min(polar) - 3, max(polar) + 3]]   # this is so that the edge of the contours aren't cut off
    density, equatedges, polaredges = np.histogram2d(equat, polar, bins=150, range=extent)
    equatbins = equatedges[:-1] + (equatedges[1] - equatedges[0]) / 2   # this fixes the order of the bins, and centers the bins at the midpoint
    polarbins = polaredges[:-1] + (polaredges[1] - polaredges[0]) / 2
    
    density = density.T      # take the transpose of the density matrix
    density = scipy.ndimage.zoom(density, 2)    # this smooths out the data so that it's less boxy and more curvey
    equatbins = scipy.ndimage.zoom(equatbins, 2)
    polarbins = scipy.ndimage.zoom(polarbins, 2)
    # density = scipy.ndimage.gaussian_filter(density, sigma=1)  # this smooths the area density even moreso (not necessary, but keeping for posterity)
    
    ax.pcolormesh(equatbins, polarbins, density, cmap=colour)     # plot the radio contours
# ax.set_ylim(0, 180); ax.set_xlim(0, 360)
ax.invert_yaxis();
ax.set_facecolor('k')
ax.set_aspect('equal')




