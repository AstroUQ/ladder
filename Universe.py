# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 13:08:36 2022

@author: ryanw
"""
import numpy as np
from tqdm import tqdm     # this is a progress bar for a for loop
from GalaxyCluster import GalaxyCluster

class Universe(object):
    def __init__(self, radius, clusters, complexity="Normal", homogeneous=False):
        '''
        '''
        self.clusterpop = clusters
        self.radius = radius
        self.complexity = complexity
        self.homogeneous = homogeneous
        self.clusters = self.generate_clusters()
        self.galaxies = self.get_all_galaxies()
        self.supernovae = self.explode_supernovae(min(55, len(self.galaxies)))
    
    def generate_clusters(self):
        '''
        '''
        threshold = 100000
        population = self.clusterpop
        clusters = []

        equat = np.random.uniform(0, 360, population)
        polar = np.random.uniform(0, 180, population)
        
        lowerbound = 2000 / self.radius     # we want a certain area around the origin to be empty (to make space for the local cluster)
        if self.homogeneous:
            dists = np.random.uniform(lowerbound, 1, population)
        else:
            median = threshold / self.radius    # we want half of the galaxies to be resolved, half to not be
            mean = median / np.log(2)       #  the mean of the exponential distribution is = median / ln(2)
            dists = np.random.exponential(mean, population) + lowerbound    # we don't want galaxy clusters within the lowerbounded sphere
        R = self.radius * np.cbrt(dists)
        
        populations = np.random.exponential(8, population)
        populations = [1 if pop < 1 else int(pop) for pop in populations]
        
        localequat = np.random.uniform(0, 360); localpolar = np.random.uniform(45, 135)

        for i in tqdm(range(self.clusterpop)):
            pos = (equat[i], polar[i], R[i])
            if i == self.clusterpop - 1:
                clusters.append(GalaxyCluster((localequat, localpolar, 2000), 15, local=True, complexity=self.complexity))
            elif R[i] > threshold:
                clusters.append(GalaxyCluster(pos, populations[i], complexity="Distant"))
            else:
                clusters.append(GalaxyCluster(pos, populations[i], complexity=self.complexity))

        return clusters
    
    def get_all_galaxies(self):
        '''
        '''
        clusters = [cluster.galaxies for cluster in self.clusters]
        flatgalaxies = [galaxy for cluster in clusters for galaxy in cluster]
        return flatgalaxies
    
    def explode_supernovae(self, frequency):
        '''
        Parameters
        ----------
        frequency : int
            The number of supernovae to generate
        '''
        indexes = np.random.uniform(0, len(self.galaxies) - 1, frequency - 2)
        closeindexes = len(self.galaxies) - np.random.uniform(1, 14, 2)
        indexes = np.append(indexes, closeindexes); np.random.shuffle(indexes)
        galaxies = [self.galaxies[int(i)] for i in indexes]
        positions = np.array([galaxy.spherical for galaxy in galaxies])
        intrinsic = 1.5 * 10**44 / (4 * np.pi * (7 * 10**6)**2)     # rough energy release of R=7000km white dwarf Type Ia supernova (W/m^2)
        distances = positions[:, 2] * 3.086 * 10**16    # convert from parsec to meters
        peakfluxes = (intrinsic / distances**2) * np.random.normal(1, 0.01, frequency)
        skypositions = [positions[:, 0] * np.random.normal(1, 0.01, frequency), 
                        positions[:, 1] * np.random.normal(1, 0.01, frequency)]   # [equat, polar]
        # print(skypositions)
        # print(fluxes)
        return skypositions, peakfluxes
        
        
        
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
            