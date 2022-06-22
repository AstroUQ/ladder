# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 17:08:36 2022

@author: ryanw
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import colors
import pandas as pd
import scipy.optimize as opt
# import colour as col

class BlackHole(object):
    def __init__(self, galaxymass, galaxytype, luminosity):
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
    
    def BH_contour(self):
        return None

def main():
    h = BlackHole(2000, "SBb", 0.8)

if __name__ == "__main__":
    main()