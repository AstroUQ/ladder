# -*- coding: utf-8 -*-
"""
Created on Fri Jan 13 13:36:37 2023

@author: ryanw
"""
import numpy as np
import MiscTools as misc
from BlackHole import BlackHole

class DistantGalaxy(object):
    
    def __init__(self, species, position, cartesian=False, blackhole=True, darkmatter=True, rotate=True):
        ''' A 'distant galaxy' which attempts to emulate many properties of a Galaxy object as a single data point.
        Parameters
        ----------
        species : str
            The Hubble classification of the galaxy. E0-7, S/Ba-c
        position : 3-tuple/list/np.array
            if cartesian == False, position = [equatorial angle, polar angle, radius (distance away)]
            if cartesian == True, position = [x, y, z]
        cartesian : bool
            Whether the provided position is in 3D cartesian coordinates (True) or spherical coordinates (False)
        blackhole : bool
            Whether or not to generate a BlackHole object at the galaxy's center
        darkmatter : bool
            Whether or not to generate dark matter in the galaxy mass. For distant galaxies, this impacts the galaxy mass
            and hence the rotation curve of the host galaxy *cluster*
        rotate : bool
            Whether or not to rotate the galaxy randomly. Value doesn't matter for distant galaxies.
        '''
        self.species = species
        self.darkmatter = darkmatter
        self.population = self.determine_population(self.species)
        self.radius = self.determine_radius(self.species)
        self.blackhole = blackhole
        self.rotate = rotate
        self.complexity = "Distant"
        if cartesian:
            self.cartesian = position
            self.spherical = misc.cartesian_to_spherical(position[0], position[1], position[2])
        else:
            self.spherical = position
            self.cartesian = misc.spherical_to_cartesian(position[0], position[1], position[2])
        self.rotation = np.array([0, 0, 0])
        self.galaxymass = self.gen_mass()
        self.bandlumin = False # set as false so that we don't need to generate it until we need to
        self.blackhole = self.generate_BlackHole()
        
    def gen_mass(self):
        ''' Generate masses distant galaxies. Lookup data taken from "Galaxy Averages.txt"
        Returns
        -------
        mass : float
            The mass of the galaxy in solar masses
        '''
        mass = 0
        if self.species[0] == "S":
            speciesindex = {"S0":0, "Sa":1, "Sb":2, "Sc":3, "SBa":4, "SBb": 5, "SBc":6}
            # [disk, bulge, bar, young spiral, old spiral] populations as a proportion of total galaxy star population
            regionpops = [[0.7, 0.2, 0, 0.01, 0.09],           #S0
                           [0.45, 0.2, 0, 0.15, 0.2],        #Sa
                           [0.45, 0.2, 0, 0.15, 0.2],        #Sb
                           [0.5, 0.2, 0, 0.1, 0.2],          #Sc
                           [0.3, 0.15, 0.2, 0.15, 0.2],      #SBa
                           [0.25, 0.15, 0.25, 0.15, 0.2],    #SBb
                           [0.4, 0.2, 0.2, 0.1, 0.1]]        #SBc
            props = regionpops[speciesindex[self.species]]
            
            # elif species == "disk":
            mass += sum(np.random.gamma(1, 0.5, int(props[0] * self.population)) + 0.08)
            # bulge and bar
            mass += sum(np.random.gamma(1, 1, int(props[1] + props[2] * self.population)) + 0.08)
            # if species in ("youngspiral", "ys"):
            mass += sum(np.random.gamma(2, 2.5, int(props[3] * self.population)) + 0.08)
            # elif species in ("oldspiral", "os"):
            mass += sum(np.random.gamma(1, 2, int(props[4] * self.population)) + 0.08)
        else: # elliptical/cD galaxy
            # approximate the stars as mostly bulge stars, somewhat as disk stars
            mass += sum(np.random.gamma(1, 0.8, self.population) + 0.08)
        
        return mass
    
    def get_bandlumin(self, evolution=True, universe_rad=450000):
        ''' Calculates and returns the specific luminosity of the distant galaxy in 3 bands (440nm, 500nm and 700nm 
            respectively).
        Parameters
        ----------
        evolution : bool
            If true, blueshifts the planck spectrum of the galaxy to emulate galaxy evolution (younger galaxies being bluer)
        universe_rad : float
            The radius of the universe (in pc). Only relevant if evolution==True, in which case it serves as a way to limit the 
            maximum blueshift of the most distant galaxies.
        Returns
        -------
        bandlumin : np.array
            The specific *luminosity* of the distant galaxy in the BGR bands respectively
            i.e. at 440nm, 500nm and 700nm. 
        '''
        # the below values are averages taken from 40 generations of full size galaxies. Not used anymore, but
        # shouldn't be deleted in case they need to be used later.
        # bandluminlookup = {"S0":[9.18e+27, 7.11e+27, 3.24e+27], "Sa":[1.74e+28, 1.30e+28, 5.44e+27], "Sb":[1.25e+28, 9.71e+27, 4.41e+27],
        #                    "Sc":[1.26e+28, 9.59e+27, 4.23e+27], "SBa":[1.33e+28, 1.03e+28, 4.64e+27], "SBb":[1.42e+28, 1.11e+28, 5.11e+27], 
        #                    "SBc":[1.26e+28, 9.77e+27, 4.43e+27], "cD":[3.35e+28, 2.34e+28, 8.69e+27], "E0":[8.44e+27, 6.53e+27, 2.96e+27], 
        #                    "E1":[9.37e+27, 7.24e+27, 3.27e+27], "E2":[8.68e+27, 6.75e+27, 3.10e+27], "E3":[1.02e+28, 7.88e+27, 3.58e+27], 
        #                    "E4":[9.01e+27, 7.00e+27, 3.21e+27], "E5":[8.46e+27, 6.51e+27, 2.91e+27], 
        #                    "E6":[9.14e+27, 7.12e+27, 3.27e+27], "E7":[9.09e+27, 7.07e+27, 3.24e+27]}
        # bandlumin = np.array(bandluminlookup[self.species]) * np.random.normal(1, 0.15, 3)
        if self.bandlumin == False: # if the band luminosities haven't been generated yet
            # these are parameters for a surge function (approximating a planck curve) based on the above data
            bandlumin_coeffs = {"S0":[1.29e+37, 1.25, -6.3e+6], "Sa":[5.02e+37, 1.28, -6.87e+6], 
                                "Sb":[1.947e+37, 1.257, -6.31e+6], "Sc":[2.2e+37, 1.258, -6.53e+6], 
                                "SBa":[2.22e+37, 1.26, -6.36e+6], "SBb":[2.02e+37, 1.254, -6.22e+6], 
                                "SBc":[1.94e+37, 1.256, -6.33e+6], 
                                "cD":[2.93e+38, 1.33, -7.8e+6], 
                                "E0":[1.26e+37, 1.254, -6.33e+6], "E1":[1.448e+37, 1.255, -6.36e+6], 
                                "E2":[1.165e+37, 1.249, -6.25e+6], "E3":[1.44e+37, 1.25, -6.33e+6], 
                                "E4":[1.217e+37, 1.249, -6.261e+6], "E5":[1.425e+37, 1.259, -6.424e+6], 
                                "E6":[1.264e+37, 1.251, -6.246e+6], "E7":[1.263e+37, 1.25, -6.26e+6]}
            
            a, b, c = bandlumin_coeffs[self.species] # obtain the curve params
            if evolution:
                d = - (self.spherical[2] / universe_rad) * 100 # calculate a horizontal shift to make lumins bluer
            else:
                d = 0 # if no evolution, we don't need a horizontal shift
            # surge function approximating a planck curve 
            surge_lumin = lambda x: a * ((x + d) * 10**-9)**b * np.exp(c * ((x + d) * 10**-9))
            
            bands = np.array([440, 500, 700])
            self.bandlumin = surge_lumin(bands) * np.random.normal(1, 0.05, 3) # calculate the lumins, with some noise too
            return self.bandlumin
        else: # if the lumins have already been generated, just return them again
            return self.bandlumin
    
    def generate_BlackHole(self):
        ''' Generates (or not!) a black hole based on the type of galaxy and complexity of the universe.
        Returns
        -------
        bh : BlackHole object
            The blackhole at the center of the galaxy
        '''
        if self.blackhole == True:
            # we want ellipticals to have more active AGN on average
            eddlumin = np.random.uniform(0.2, 1) if self.species[0] == "S" else np.random.uniform(0.5, 1)
            bh = BlackHole(self.galaxymass / 2, self.species, self.radius, eddlumin)
        else:
            bh = False
        return bh
    
    def determine_population(self, species):
        ''' Determine the number of stars to put in a galaxy depending on the galaxy type
        Parameters
        ----------
        species : str
            One of {cD, E0-7, S0, Sa, Sb, Sc, SBa, SBb, SBc} as per the galaxy type. 
        Returns
        -------
        population : int
            The number of stars to generate in the galaxy
        '''
        # first, get a key based on the species of the galaxy
        if species[0] == "E":
            num = float(species[1])
            index = "E"
        else:
            num = 0
            index = species
        # now use that key to find the mean and standard dev of a normal dist. for the stellar population
        poplookup = {"S0":[1100, 50], "Sa":[1000, 100], "Sb":[900, 100], "Sc":[800, 80],
                      "SBa":[1100, 100], "SBb":[1000, 100], "SBc":[900, 80],
                      "cD":[2000, 200], "E":[1600 - 120 * num, 200 / (num + 1)]}
        mean, SD = poplookup[index]
        population = np.random.normal(mean, SD)
        return int(population)
    
    def determine_radius(self, species):
        ''' Determine the radius of a galaxy depending on the galaxy type
        Parameters
        ----------
        species : str
            One of {cD, E0-7, S0, Sa, Sb, Sc, SBa, SBb, SBc} as per the galaxy type. 
        Returns
        -------
        radius : float
            The radius of the galaxy in pc
        '''
        # first, get a key based on the species of the galaxy
        if self.species[0] == "E":
            num = float(species[1])
            index = "E"
        else:
            num = 0
            index = self.species
        # now use that key to find the mean and standard dev of a normal dist. for the radius
        radlookup = {"S0":[90, 8], "Sa":[80, 8], "Sb":[75, 6], "Sc":[70, 5],
                      "SBa":[90, 10], "SBb":[85, 7], "SBc":[75, 5],
                      "cD":[200, 50], "E":[200 - 20 * num, 40 / (num + 1)]}
        mean, SD = radlookup[index]
        radius = np.random.normal(mean, SD)
        return abs(radius)