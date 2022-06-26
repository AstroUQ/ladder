# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 08:38:00 2022

@author: ryanw
"""

import numpy as np

class BlackHole(object):
    def __init__(self, galaxymass, galaxytype, galaxyradius, luminosity):
        '''
        Parameters
        ----------
        galaxymass : float
            baryonic mass of the galaxy in solar masses
        luminosity : float
            the fraction of the eddington luminosity that the black hole should be
        '''
        self.mass = self.initialise_mass(galaxymass)
        self.luminosity = luminosity * self.eddington_luminosity(self.mass)
        self.galaxyradius = galaxyradius
        if galaxytype[0] == "S":
            self.BHradio = self.BH_emission(FR=1)
        else:
            self.BHradio = self.BH_emission(FR=2)
    
    def initialise_mass(self, galaxymass):
        return (galaxymass * 3 * 10**-2) * np.random.normal(1, 0.1)
    
    def eddington_luminosity(self, mass):
        ''' Eddington luminosity for an accreting black hole. 
        '''
        return 3 * 10**4 * (mass)
    
    def get_BH_mass(self):
        return self.mass
    def get_BH_lumin(self):
        return self.luminosity
    def get_BH_colour(self):
        ''' Quasar RGB colour. 
        '''
        return np.array([73, 214, 255]) / 255
    def get_BH_radio(self):
        return self.BHradio
    
    def get_BH_scale(self):
        '''A basic logarithmic scale to determine how large stars should be in pictures given their luminosity.
        Returns
        -------
        scale : float
            the number to be input into the 'scale' kwarg in a matplotlib figure. 
        '''
        scale = 3 * np.log(2 * self.luminosity + 1)
        scale = 2 if scale > 2 else scale
        return scale
    
    def BH_emission(self, FR=2):
        ''' Produce mock-radio emission lobes from the SMBH in the center of the galaxy according to its Fanaroff-Riley type, FR. 
        Inspiration for shaping came from  A.H. Bridle et al. 1994, Deep VLA imaging of twelve extended 3CR quasars, AJ 108, 766,
        available at : https://articles.adsabs.harvard.edu/pdf/1994AJ....108..766B
        This function emulates emission lobes by randomly scattering dots in 3D space according to rules and specified positions:
            1. A central core, with 'centerpop' dots and radius 0.1pc
            2. Jets (non-symmetric if FR-2), extending out of the galactic plane
            3. Lobes, which expand out from the ends of the jet. 
        This function purely creates the dots. It is the job of other functions (Galaxy.galaxy_radio) to plot the area density ("intensity")
        Parameters
        ----------
        FR : int
            {1, 2} depending on which Fanaroff-Riley emission type to simulate. 
        '''
        if FR == 1:
            centerpop = 500
            centerradius = 0.1
            # the following distributes points more or less evenly about a sphere centered at the origin
            theta = np.random.uniform(0, 2*np.pi, centerpop)
            phi = np.random.uniform(-1, 1, centerpop)
            phi = np.arccos(phi)
            centerx = centerradius * (np.cos(theta) * np.sin(phi) * np.random.normal(1, 0.1, centerpop))
            centery = centerradius * (np.sin(theta) * np.sin(phi) * np.random.normal(1, 0.1, centerpop))
            centerz = centerradius * (np.cos(phi) * np.random.normal(1, 0.3, centerpop))
            
            jetpop = 1000
            jetradius = 0.5 * self.galaxyradius
            jetz = jetradius * (np.geomspace(0.01, 1.6, jetpop) * np.random.normal(1, 0.01, jetpop))    
            jetx = np.random.normal(0, 0.2 * jetz, jetpop)
            jety = np.random.normal(0, 0.2 * jetz, jetpop)
            
            jetreflect = np.random.uniform(0, 1, jetpop)
            for i, val in enumerate(jetreflect):
                if val > 0.5:         # half of the points are reflected (symmetrical)
                    jetz[i] *= -1; jetx[i] *= -1; jety[i] *= -1
                    
            lobepop = 4000
            loberadius = jetradius * 2
            mult = 5    # arbitrary divisor to compact the lobes towards the jet a little bit
            lobeangle = np.geomspace(0.5 * np.pi, 1.5 * np.pi, lobepop)
            lobex = loberadius / mult * (lobeangle * np.cos(1.2 * lobeangle) * np.random.normal(1, 0.1, lobepop) + np.random.normal(0, 0.3 * lobeangle, lobepop))
            lobey = loberadius / mult * (lobeangle * np.sin(1.2 * lobeangle) * np.random.normal(1, 0.1, lobepop) + np.random.normal(0, 0.3 * lobeangle, lobepop)) - jetradius/2
            lobez = 0.5 * jetradius + (loberadius / mult * (lobeangle * np.sin(0.7 * lobeangle) * np.random.normal(1, 0.2, lobepop) + np.random.normal(0, 0.2 * lobeangle, lobepop)))
            
            lobereflect = np.random.uniform(0, 1, lobepop)
            for i, val in enumerate(lobereflect):
                if val > 0.5:       # half of the points are reflected
                    lobez[i] *= -1; lobex[i] *= -1; lobey[i] *= -1  # reflect the coordinates about their respective axes. 
        # the next section is functionally identical - i couldn't really be bothered reducing the linecount here
        else:
            centerpop = 100
            centerradius = 0.1
            theta = np.random.uniform(0, 2*np.pi, centerpop)
            phi = np.random.uniform(-1, 1, centerpop)
            phi = np.arccos(phi)
            centerx = centerradius * (np.cos(theta) * np.sin(phi) * np.random.normal(1, 0.1, centerpop))
            centery = centerradius * (np.sin(theta) * np.sin(phi) * np.random.normal(1, 0.1, centerpop))
            centerz = centerradius * (np.cos(phi) * np.random.normal(1, 0.3, centerpop))
            
            jetpop = 1000
            jetradius = 2.5 * self.galaxyradius
            jetz = jetradius * (np.linspace(0.01, 1, jetpop) * np.random.normal(1, 0.01, jetpop))
            jetx = np.random.normal(0, 0.01 * jetz, jetpop)
            jety = np.random.normal(0, 0.01 * jetz, jetpop)
            
            jetreflect = np.random.uniform(0, 1, jetpop)
            for i, val in enumerate(jetreflect):
                if val > 0.9:       # only about 10% of the points are reflected
                    jetz[i] *= -1; jetx[i] *= -1; jety[i] *= -1
            
            lobepop = 4000
            loberadius = jetradius * 2/3
            mult = 5    # arbitrary divisor to compact the lobes towards the jet a little bit
            lobeangle = np.geomspace(0.5 * np.pi, 1 * np.pi, lobepop)
            lobex = loberadius / mult * (lobeangle * np.cos(lobeangle) * np.random.normal(1, 0.1, lobepop) + np.random.normal(0, 0.2 * lobeangle, lobepop))# * reflect 
            lobey = loberadius / mult * (np.random.normal(0, 0.1, lobepop) + np.random.normal(0, 0.2 * lobeangle, lobepop))# * - reflect + np.random.normal(0, scatter2, lobepop))
            lobez = 0.7 * jetradius + (loberadius / mult * (lobeangle * np.sin(lobeangle) * np.random.normal(1, 0.2, lobepop) + np.random.normal(0, 0.2 * lobeangle, lobepop)))
            
            lobereflect = np.random.uniform(0, 1, lobepop)
            for i, val in enumerate(lobereflect):
                if val > 0.6:
                    lobez[i] *= -1; lobex[i] *= -1; lobey[i] *= -1
            
        x = np.concatenate((centerx, jetx, lobex), axis=0)
        y = np.concatenate((centery, jety, lobey), axis=0)
        z = np.concatenate((centerz, jetz, lobez), axis=0)
        radius = centerradius + jetradius + loberadius
        return x, y, z, radius