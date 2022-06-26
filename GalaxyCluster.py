# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 08:38:20 2022

@author: ryanw
"""
import numpy as np
from multiprocessing import Pool
from Galaxy import Galaxy

class GalaxyCluster(object):
    def __init__(self, position, population, cartesian=False, local=False):
        '''
        Parameters
        ----------
        position : 3-tuple/list/np.array
            if cartesian == False, position = [equatorial angle, polar angle, radius (distance away)]
            if cartesian == True, position = [x, y, z]
        local : bool
            Whether this is the local galaxy cluster (i.e. the one that the observer at the origin is in)
        '''
        self.local = local
        self.radius = 1000
        if cartesian:
            self.cartesian = position
            self.spherical = self.cartesian_to_spherical(position[0], position[1], position[2])
        else:
            self.spherical = position
            self.cartesian = self.spherical_to_cartesian(position[0], position[1], position[2])
        self.galaxies = self.generate_galaxies(population)
    
    def generate_galaxy(self, species, position, population, radius):
        return Galaxy(species, position, population, radius, cartesian=True)
    
    def generate_galaxies(self, population):
        theta = np.random.uniform(0, 2*np.pi, population)
        phi = np.random.uniform(-1, 1, population)
        phi = np.arccos(phi)
        
        dists = np.random.exponential(0.4, population)
        R = self.radius * dists**(1/3)
        
        x = R * (np.cos(theta) * np.sin(phi) + np.random.normal(0, 0.1, population)) + self.cartesian[0]
        y = R * (np.sin(theta) * np.sin(phi) + np.random.normal(0, 0.1, population)) + self.cartesian[1]
        z = R * (np.cos(phi) + np.random.normal(0, 0.05, population)) + self.cartesian[2]
        
        args = [('Sa', [x[i], y[i], z[i]], 600, 70) for i in range(len(x))]
        print(args)
        pool = Pool()
        galaxies = pool.starmap(self.generate_galaxy, args)
        pool.close()
        pool.join()
        # args = [['Sa', [x[i], y[i], z[i]], 500, 70] for i in range(len(x))]
        # galaxies = []
        # for i in range(len(x)):
        #     species, position, population, radius = args[i]
        #     print(species, position, population, radius)
        #     galaxies.append(Galaxy(species, position, population, radius, cartesian=True))
        return galaxies
            
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
