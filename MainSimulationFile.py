# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 09:52:26 2022

@author: ryanw

"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# import colour as col

class Star(object):
    def __init__(self, location, species="MS"):
        self.colourdata = pd.read_csv("blackbodycolours.txt", delimiter=' ')
        if species == "MS":
            self.mass = abs(self.MS_masses(location))
            self.luminosity = abs(self.MS_lumin(self.mass))
            self.radius = self.MS_radius(self.mass)
            self.temperature = self.MS_temperature(self.luminosity, self.radius)
        elif species == "WDwarf":
            self.mass = abs(self.WD_masses())
            self.radius = self.WD_radii(self.mass)
            self.temperature = self.WD_temp(self.mass)
            self.luminosity = abs(self.WD_lumin(self.temperature, self.radius, self.mass))
        elif species == "Giant":
            self.temperature = self.giant_temp()
            self.luminosity = abs(self.giant_lumin(self.temperature))
            self.mass = abs(self.giant_mass())
            self.radius = self.stefboltz_radius(self.luminosity, self.temperature)
        elif species == "SupGiant":
            self.temperature = self.SGB_temp()
            self.luminosity = abs(self.SGB_lumin(self.temperature))
            self.mass = abs(self.SGB_mass())
            self.radius = self.stefboltz_radius(self.luminosity, self.temperature)
            
    def MS_masses(self, species):
        '''Masses for stars on the main sequence.
        Parameters
        ----------
        species : str
            which part of the galaxy the star is in.
        '''
        if species in ("youngspiral", "ys"):
            a, b = 2, 2.5
        elif species in ("oldspiral", "os"):
            a, b = 1, 2
        elif species == "disk":
            a, b = 1, 0.5
        elif species == "bulge":
            a, b = 1, 1
        return np.random.gamma(a, b) + 0.08
    
    def MS_lumin(self, mass):
        '''Piecewise relationships taken from:
            Wikipedia : https://en.wikipedia.org/wiki/Mass%E2%80%93luminosity_relation
            Evolution of Stars and Stellar Populations (book) : https://cfas.org/data/uploads/astronomy-ebooks/evolution_of_stars_and_stellar_populations.pdf
        Parameters
        ----------
        mass : float
            solar masses of the star
        Returns
        -------
        lumin : float
            solar luminosities of the star
        '''
        if 0.08 <= mass <= 0.43:
            lumin = 0.23 * mass**2.3
        elif 0.43 < mass <= 2:
            lumin = mass**4
        elif 2 < mass <= 55:
            lumin = 1.4 * mass**3.5
        elif 55 < mass:
            lumin = 3.2 * 10**4 * mass
        else: ValueError("something is wrong")
        lumin += np.random.normal(0, 0.005 * mass)
        lumin = 0.23 * 0.04**2.3 if lumin < 0.23 * 0.04**2.3 else lumin
        return lumin

    def MS_radius(self, mass):
        '''Piecewise relationships taken from:
            https://articles.adsabs.harvard.edu/pdf/1991Ap%26SS.181..313D
            (with some tweaking)
        
        ZAMS = zero-age main sequence (new star)
        TAMS = terminal age main sequence (about to transition off of MS)
        '''
        if mass < 1.66:
            TAMS = 2 * mass**0.75
            ZAMS = 0.89 * mass**0.89
        elif mass >= 1.66:
            TAMS = 1.61 * mass**0.83
            ZAMS = 1.31 * mass**0.57
        ave = (TAMS + ZAMS) / 2
        radius = np.random.normal(ave, 0.12 * mass)
        return radius
        
    def MS_temperature(self, lumin, radius):
        '''Main sequence temperature using stefan-boltzman equation. 
        '''
        sigma = 5.67037 * 10**-8
        R = 696340000 * radius
        L = 3.828 * 10**26 * lumin
        temp = (L / (sigma * 4 * np.pi * R**2 ))**(1/4)
        temp += np.random.normal(0, 0.1 * temp)
        temp = 40000 if temp > 40000 else temp
        return temp
    
    def WD_masses(self):
        '''Masses for white dwarf stars.
        Strong peak at M ~ 0.65, and must be in the range [0.17, 1.33]
        Distribution retrieved from Fig 1 from:
            https://www.lume.ufrgs.br/bitstream/handle/10183/90266/000586456.pdf?sequence=1
        Four different normal distributions make up the total distribution, with mass fractions 7%, 69%, 23% and 1% respectively. 
        
        Parameters
        ----------
        n : int
            the number of white dwarfs to simulate
        
        Returns
        -------
        masses : np.array or float64
            The mass of the white dwarf stars in units Solar Masses
        '''
        prob = np.random.uniform(0, 1)
        if prob <= 0.07:
            mass = np.random.normal(0.38, 0.05)
        elif prob <= 0.69+0.07:
            mass = np.random.normal(0.578, 0.05)
        elif prob <= 0.23 + 0.69 + 0.07:
            mass = np.random.normal(0.658, 0.2)
        else:
            mass = np.random.normal(1.081, 0.2)
        mass = 1.33 if mass > 1.33 else mass
        mass = 0.17 if mass < 0.17 else mass
        return mass
    
    def WD_radii(self, mass):
        '''Radius ~ M^(-1/3) from wikipedia :
            https://en.wikipedia.org/wiki/White_dwarf
        '''
        radii = 6 * 10**-3 * mass**(-1/3) * np.random.normal(1, 0.01)
        return radii

    def WD_temp(self, mass):
        '''I don't remember where this came from -- i think I may have curve-fit wikipedia data on a log scale. 
        '''
        return 10**(0.7 * np.log10(mass) + 4.4) * np.random.normal(1, 0.05)

    def WD_lumin(self, temps, radii, masses):
        ''' Uses the Stefan-Boltzmann equation with some multiplier to calculate lumin of white dwarf. 
        Parameters
        ----------
        masses : float or np.array
            masses of the white dwarf stars in units of solar masses
            
        Returns
        -------
        luminosity : float
            Luminosity of the white dwarf in solar luminosities
        '''
        sigma = 5.67037 * 10**-8
        R = 696340000 * radii
        solLum = 3.828 * 10**26
        mult =  1 / solLum
        return 4 * np.pi * R**2 * sigma * temps**4 * mult * np.random.normal(1, 0.1)

    def SGB_temp(self):
        '''Beta temperature distribution, weighted to be just higher in temperature than the midpoint. 
        Returns
        -------
        temp : float
            temp in Kelvin. 
        '''
        a, b = 2.5, 2
        mintemp, maxtemp = 2000, 2.2 * 10**4
        temp = (np.random.beta(a, b) * maxtemp) + mintemp
        return temp * np.random.normal(1, 0.2)

    def SGB_lumin(self, temp):
        ''' Modelled to be a inverted parabola, given start and end points at the low and high temperature extrema,
        and to be around 10^4.5 solar luminosities. 
        Returns
        -------
        luminosity : float
            solar luminosities of the supergiant star. 
        '''
        a, b, c = -0.0076, 206.516, -350972
        lumin = a * temp**2 + b * temp + c
        return lumin * np.random.normal(1, 0.3)
    
    def SGB_mass(self):
        ''' Normally distributed mass centered at 20 solar masses with SD of 8. 
        '''
        return np.random.normal(10, 8) + 10

    def giant_temp(self):
        ''' Gamma distribution of star temperatures, weighted to be lower (~4000). 
        '''
        a, b = 4, 1
        temp = 1000 * np.random.gamma(a, b) + 2000
        return temp * np.random.normal(1, 0.1)

    def giant_lumin(self, temp):
        ''' Not entirely sure what's going on here. Returns luminosity in solar luminosities. 
        '''
        a, b, c = 10**-7 * 2.857, -0.00343, 10.71
        add = 10**4.3 * np.exp(a*(temp-4000)**2 + b*temp + c) #what the heck is this (it adds extra luminosity to low temp stars)
        return (temp + add) * abs(np.random.normal(1, 0.8)) * 0.02
    
    def giant_mass(self):
        ''' Normally distributed mass centered at 8 solar masses with SD of 3. 
        '''
        return np.random.normal(4, 3) + 4 
    
    def stefboltz_radius(self, lumin, temp):
        ''' Stefan-Boltzman equation to calculate radius. 
        Parameters
        ----------
        lumin, temp : float
            the solar luminosities and temp (K) of the star
        Returns
        -------
        radius : float
            solar radii of the star. 
        '''
        sigma = 5.67037 * 10**-8
        lumin = 3.828 * 10**26 * lumin
        radius = np.sqrt(lumin / (4 * np.pi * sigma * temp**4))
        return radius / 696340000

    def get_star_colour(self):
        ''' Takes the RGB colour values from a blackbody of specified temperature, with values stored in the file:
            "blackbodycolours.txt"
        Blackbody RGB values kindly supplied by:
            Mitchell Charity <mcharity@lcs.mit.edu>
            http://www.vendian.org/mncharity/dir3/blackbody/
            Version 2001-Jun-22
        
        Alternate version (commented out) using colour-science package:
            Approximations were retrieved from https://en.wikipedia.org/wiki/Planckian_locus
        '''
        #i tried to use an algorithm here but i ran into issues with the colour-science (or is it colour?) package
        # temp = self.temperature
        # if 1667 <= temp <= 4000:
        #     x = -0.2661239 * (10**9 / temp**3) - 0.2343589 * (10**6 / temp**2) + 0.8776956 * (10**3 / temp) + 0.179910
        #     if temp <= 2222:
        #         y = -1.1063814 * x**3 - 1.34811020 * x**2 + 2.18555832 * x - 0.20219683
        #     else:
        #         y = -0.9549476 * x**3 - 1.37418593 * x**2 + 2.09137015 * x - 0.16748867
        # elif 4000 <= temp <= 25000:
        #     x = -3.0258469 * (10**9 / temp**3) + 2.1070379 * (10**6 / temp**2) + 0.2226347 * (10**3 / temp) + 0.24039
        #     y = 3.0817580 * x**3 - 5.87338670 * x**2 + 3.75112997 * x - 0.37001483
        # xy = [x, y]
        # XYZ = col.xy_to_XYZ(xy)
        # rgb = col.XYZ_to_RGB(XYZ)
        # return rgb
        temperature = self.temperature
        temperature = 40000 if temperature > 40000 else temperature
        temperature = 1000 if temperature < 1000 else temperature
        temp = round(temperature / 100) * 100
        r, g, b = self.colourdata.loc[self.colourdata['Temperature'] == temp].iloc[0, 9:12]
        rgb = np.array([r, g, b]) / 255
        return rgb
    
    def get_star_scale(self):
        '''A basic logarithmic scale to determine how large stars should be in pictures given their luminosity.
        Returns
        -------
        scale : float
            the number to be input into the 'scale' kwarg in a matplotlib figure. 
        '''
        scale = 3 * np.log(2 * self.luminosity + 1)
        scale = 2 if scale > 2 else scale
        return scale
    
    def get_star_lumin(self):
        return self.luminosity
    def get_star_mass(self):
        return self.mass
    def get_star_temp(self):
        return self.temperature
    def get_star_radius(self):
        return self.radius
    
        

class Galaxy(object):
    def __init__(self, species, position, population, radius, cartesian=False):
        '''
        Parameters
        ----------
        species : str
        position : 3-tuple/list/np.array
            if cartesian == False, position = [equatorial angle, polar angle, radius (distance away)]
            if cartesian == True, position = [x, y, z]
        '''
        self.species = species
        self.population = population
        self.radius = radius
        if cartesian:
            self.cartesian = position
            self.spherical = self.cartesian_to_spherical(position[0], position[1], position[2])
        else:
            self.spherical = position
            self.cartesian = self.spherical_to_cartesian(position[0], position[1], position[2])
        self.starpositions, self.stars = self.generate_galaxy()
        
    def galaxyrotation(self, angle, axis):
        '''Rotate a point in cartesian coordinates about the origin by some angle along the specified axis. 
        The rotation matrices were taken from https://stackoverflow.com/questions/34050929/3d-point-rotation-algorithm
        
        Parameters
        ----------
        angle : float
            An angle in radians.
        axis : string
            The axis to perform the rotation on. Must be in ['x', 'y', 'z'].
        
        Returns
        -------
        numpy array
            The transformation matrix for the rotation of angle 'angle'. This output must be used as the first argument within "np.dot(a, b)"
            where 'b' is an 3 dimensional array of coordinates.
        '''
        if axis == 'x':
            m = np.array([[1, 0, 0], [0, np.cos(angle), -np.sin(angle)], [0, np.sin(angle), np.cos(angle)]])
            return m
        elif axis == 'y':
            m = np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]])
            return m
        else:
            m = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
            return m
    
    def generate_spiral(self, population, radius):
        '''Barred spiral galaxies generated using Fermat's spiral, and standard spiral galaxies generated using Archimedean spiral. 
        Returns
        -------
        x, y, z : np.array (x3)
            cartesian coordinates of each star in the galaxy
        colours : np.array
            Each element is an [R, G, B] value of each star to put into a matplotlib figure
        scales : np.array
            an array of floats which dictates how large a star appears on a matplotlib figure
        stars : np.array
            an array of Star objects, for each star in the galaxy
        '''
        # first step is to define the index to use in multiple data tables, based on the galaxy type:
        speciesindex = {"S0":0, "Sa":1, "Sb":2, "Sc":3, "SBa":4, "SBb": 5, "SBc":6}
        # galaxy radii need to be divided by certain numbers so that they behave as expected, with the divisor dependent on galaxy type
        SpiralRadiiDiv = [None, 15, 7, 2.1, 3.7, 3, 2.3]      #radius divisors (unitless)
        # for barred galaxies, the extent of the bar is different for different types of galaxy
        barradii = [0, 0, 0, 0, 0.3, 0.4, 0.5]  # bar radius as proportion of galaxy radius
        # next is the angular extents for each of the spiral arms in the form of [lower, upper] angle (radians)
        wrap = [[None, None], [0.9, 4 * np.pi], [0.7, 2 * np.pi], [0.2, 0.8 * np.pi], 
                [np.pi / 2.1, 3 * np.pi], [np.pi / 2.1, 2 * np.pi], [np.pi / 2.1, 1.15 * np.pi]]
        
        #now to actually grab the parameters for the galaxy type in question:
        mult, spiralwrap = [param[speciesindex[self.species]] for param in [SpiralRadiiDiv, wrap]]
        
        #[disk, bulge, bar, spiral] populations as a proportion of total galaxy star population
        regionpops = [[0.7, 0.2, 0, 0.01, 0.09],           #S0
                       [0.45, 0.2, 0, 0.15, 0.2],        #Sa
                       [0.45, 0.2, 0, 0.15, 0.2],        #Sb
                       [0.5, 0.2, 0, 0.1, 0.2],          #Sc
                       [0.3, 0.15, 0.2, 0.15, 0.2],      #SBa
                       [0.25, 0.15, 0.25, 0.15, 0.2],    #SBb
                       [0.4, 0.2, 0.2, 0.1, 0.1]]        #SBc
        
        #now to turn those population proportions into actual populations, given the current galaxy type
        diskpop, bulgepop, barpop, youngpop, oldpop = [int(prop * population) for prop in regionpops[speciesindex[self.species]]]
        spiralpop = youngpop + oldpop
        bulgeradius = radius / 10
        
        diskdists = np.random.exponential(radius / 4, size=diskpop)
        
        theta = np.random.uniform(0, 2*np.pi, diskpop)
        
        #this defines the disk star positions
        diskx = np.cos(theta) * diskdists
        disky = np.sin(theta) * diskdists
        diskz = np.zeros(diskpop) + 0.02 * radius * np.random.randn(diskpop)
        diskstars = self.generate_stars("disk", diskpop)
        diskcolours = [star.get_star_colour() for star in diskstars]
        diskscales = [star.get_star_scale() for star in diskstars]
        
        #this defines the bulge star positions
        # bulgedists = np.random.weibull(1.5 * bulgeradius, size = bulgepop) * np.random.normal(1, 0.05, bulgepop)
        bulgedists = np.random.exponential(bulgeradius/1.3, bulgepop) * np.random.normal(1, 0.05, bulgepop)
        theta = np.random.uniform(0, 2*np.pi, bulgepop)
        phi = np.random.uniform(-1, 1, bulgepop)
        phi = np.arccos(phi)
        
        bulgeR = bulgeradius * bulgedists**(1/3)    #bulgedists was meant to be RVs between 0 and 1, but the mult makes up for it
        bulgex = bulgeR * (np.cos(theta) * np.sin(phi) + np.random.normal(0, 0.1, bulgepop))
        bulgey = bulgeR * (np.sin(theta) * np.sin(phi) + np.random.normal(0, 0.1, bulgepop))
        distanceflat = (1 / radius) * np.sqrt(bulgex**2 + bulgey**2)     #this makes the z lower for stars further from the center
        bulgez = (0.83 * bulgeR * (np.cos(phi) + np.random.normal(0, 0.1, bulgepop))) * 0.9**distanceflat
        
        # bulgex = np.cos(theta) * bulgedists
        # bulgey = np.sin(theta) * bulgedists
        # bulgez = np.random.normal(0, 1.4/3 * bulgeradius, bulgepop)
        bulgestars = self.generate_stars("bulge", bulgepop)
        bulgecolours = [star.get_star_colour() for star in bulgestars]
        bulgescales = [star.get_star_scale() for star in bulgestars]
        
        if self.species[:2] == "SB":    #this will create the bar, given that the galaxy is a barred type
            barradius = barradii[speciesindex[self.species]] * radius
            barx = np.random.normal(0, 0.07 * barradius, barpop)
            bary = barradius * (np.geomspace(0.3, 1.1, barpop) * np.random.choice([-1, 1], barpop) + np.random.normal(0, 0.1, barpop))
            barz = np.random.normal(0, 0.05 * barradius, barpop)
            barstars = self.generate_stars("bulge", barpop)
            barcolours = [star.get_star_colour() for star in barstars]
            barscales = [star.get_star_scale() for star in barstars]
            bulgex = np.append(bulgex, barx); bulgey = np.append(bulgey, bary); bulgez = np.append(bulgez, barz)
            bulgecolours = np.append(bulgecolours, barcolours, axis=0)
            bulgescales = np.append(bulgescales, barscales, axis=0)
            bulgestars = np.append(bulgestars, barstars, axis=0)
        
        # initialise some lists
        spiralx, spiraly, spiralz, spiralcolours, spiralscales, spiralstars = [], [], [], np.empty((0,3)), [], []
        
        if mult != None:          # time to generate spiral structure
            lower, upper = spiralwrap
            # youngpop, oldpop = int(spiralpop / 2), int(spiralpop / 2)
            youngstars = ["ys", youngpop, 0, 0.04, 0.01, 0.005, 10000, 6000] #[pop, lag, scatter, scatter2, zscatter, tempmean, tempshift]
            oldstars = ["os", oldpop, 0.2, 0.08, 0.015, 0.01, 4000, 1000]
            spiralpopulations = [youngstars, oldstars]
            
            for [region, pop, lag, scatter, scatter2, zscatter, tempmean, tempshift] in spiralpopulations:
                if speciesindex[self.species] >= 5:
                    spiralangle = np.geomspace(lower, upper, pop)
                else:
                    spiralangle = np.linspace(lower, upper, pop)
                reflect = np.random.choice([-1, 1], pop)
                power = 1/2 if self.species[:2] == "SB" else 1
                x = (radius / mult) * (spiralangle**power * np.cos(spiralangle + lag)  * np.random.normal(1, scatter, pop) * reflect + np.random.normal(0, scatter2, pop))
                y = (radius / mult) * (spiralangle**power * np.sin(spiralangle + lag) * np.random.normal(1, scatter, pop) * - reflect + np.random.normal(0, scatter2, pop))
                z = np.random.normal(0, zscatter * radius, pop)
                stars = self.generate_stars(region, pop)
                colours = [star.get_star_colour() for star in stars]
                scales = [star.get_star_scale() for star in stars]
                spiralx = np.append(spiralx, x); spiraly = np.append(spiraly, y); spiralz = np.append(spiralz, z)
                spiralcolours = np.append(spiralcolours, colours, axis=0)
                spiralscales = np.append(spiralscales, scales, axis=0)
                spiralstars = np.append(spiralstars, stars, axis=0)
        else:
            theta = np.random.uniform(0, 2*np.pi, spiralpop)
            x = np.cos(theta) * radius/1.5 * np.random.normal(1, 0.1, spiralpop)
            y = np.sin(theta) * radius/1.5 * np.random.normal(1, 0.1, spiralpop)
            z = np.zeros(spiralpop) + 0.02 * radius * np.random.randn(spiralpop)
            stars = self.generate_stars("disk", spiralpop)
            colours = [star.get_star_colour() for star in stars]
            scales = [star.get_star_scale() for star in stars]
            spiralx = np.append(spiralx, x); spiraly = np.append(spiraly, y); spiralz = np.append(spiralz, z)
            spiralcolours = np.append(spiralcolours, colours, axis=0)
            spiralscales = np.append(spiralscales, scales, axis=0)
            spiralstars = np.append(spiralstars, stars, axis=0)
                
        x = np.append(diskx, np.append(bulgex, spiralx)); y = np.append(disky, np.append(bulgey, spiraly)); z = np.append(diskz, np.append(bulgez, spiralz))
        colours = np.append(diskcolours, np.append(bulgecolours, spiralcolours, axis=0), axis=0)
        scales = np.append(diskscales, np.append(bulgescales, spiralscales, axis=0), axis=0)
        stars = np.append(diskstars, np.append(bulgestars, spiralstars, axis=0), axis=0)
        return x, y, z, colours, scales, stars
    
    def generate_elliptical(self, population, radius):
        '''Some guidance was taken from https://itecnote.com/tecnote/python-sampling-uniformly-distributed-random-points-inside-a-spherical-volume/
        Returns
        -------
        x, y, z : np.array (x3)
            cartesian coordinates of each star in the galaxy
        colours : np.array
            Each element is an [R, G, B] value of each star to put into a matplotlib figure
        scales : np.array
            an array of floats which dictates how large a star appears on a matplotlib figure
        stars : np.array
            an array of Star objects, for each star in the galaxy
        '''
        centralpop = int(0.2 * population); spherepop = int(0.8 * population)
        
        centralradius = radius / 6
        
        ellipsoid_mult = (1 - float(self.species[1]) / 10) # this makes later type ellipticals flatter (oblate)
        
        theta = np.random.uniform(0, 2*np.pi, population)
        phi = np.random.uniform(-1, 1, population)
        phi = np.arccos(phi)
        
        spheredists = np.random.exponential(0.4, spherepop)
        centraldists = np.random.exponential(1/5, centralpop)
        centralR = centralradius * centraldists**(1/3)
        sphereR = radius * spheredists**(1/3)
        
        centralx = centralR * (np.cos(theta[:centralpop]) * np.sin(phi[:centralpop]) + np.random.normal(0, 0.1, centralpop))
        centraly = centralR * (np.sin(theta[:centralpop]) * np.sin(phi[:centralpop]) + np.random.normal(0, 0.1, centralpop))
        centralz = centralR * (np.cos(phi[:centralpop]) + np.random.normal(0, 0.05, centralpop))
        
        centralstars = self.generate_stars("bulge", centralpop)
        centralcolours = [star.get_star_colour() for star in centralstars]
        centralscales = [star.get_star_scale() for star in centralstars]
        
        spherex = sphereR * (np.cos(theta[centralpop:]) * np.sin(phi[centralpop:]) + np.random.normal(0, 0.1, spherepop))
        spherey = sphereR * (np.sin(theta[centralpop:]) * np.sin(phi[centralpop:]) + np.random.normal(0, 0.1, spherepop))
        distanceflat = (1 / radius) * np.sqrt(spherex**2 + spherey**2)
        spherez = (sphereR * (np.cos(phi[centralpop:]) + np.random.normal(0, 0.1, spherepop))) * ellipsoid_mult**distanceflat
        
        spherestars = self.generate_stars("disk", spherepop)
        spherecolours = [star.get_star_colour() for star in spherestars]
        spherescales = [star.get_star_scale() for star in spherestars]
        
        x = np.append(centralx, spherex); y = np.append(centraly, spherey); z = np.append(centralz, spherez)
        colours = np.append(centralcolours, spherecolours, axis=0)
        scales = np.append(centralscales, spherescales, axis=0)
        
        stars = np.append(centralstars, spherestars, axis=0)
        return x, y, z, colours, scales, stars
    
    def generate_galaxy(self):
        '''Generate random stars according to species type of galaxy. 
        
        Returns
        -------
        numpy array (x4)
            Cartesian coordinates [x, y, z] of each of the stars in this galaxy, as well as an array of colours for each star. 
        '''
        population, radius = self.population, self.radius
        
        if self.species[0] == 'E':  #elliptical galaxy
            x, y, z, colours, scales, stars = self.generate_elliptical(population, radius)
        else:                       #spiral galaxy
            x, y, z, colours, scales, stars = self.generate_spiral(population, radius)
        
        points = np.array([x, y, z])
        phi = np.random.uniform(0, 2*np.pi, 3)
        
        #rotate the galaxy randomly
        points = np.dot(self.galaxyrotation(phi[0], 'x'), points)
        points = np.dot(self.galaxyrotation(phi[1], 'y'), points)
        points = np.dot(self.galaxyrotation(phi[2], 'z'), points)
        x0, y0, z0 = self.cartesian
        x, y, z = points[0] + x0, points[1] + y0, points[2] + z0  #move the galaxy away from the origin to its desired position
        return [x, y, z, colours, scales], stars
        
    def get_stars(self):
        return self.starpositions
    
    def generate_stars(self, region, n):
        '''Generates n Star objects according to the region of the galaxy.
        Parameters
        ----------
        region : str
            The region of the galaxy (e.g. young spiral, bulge, etc)
        n : int
            The number of stars to generate.
        
        Returns
        -------
        stars : list of n Star objects
        '''
        proportions = {"ys":[0.82, 0.1, 0.07, 0.01],    # [Main sequence, giants, supergiants, white dwarfs]
                       "os":[0.79, 0.15, 0.03, 0.03],
                       "disk":[0.9, 0.05, 0.02, 0.03], 
                       "bulge":[0.8, 0.1, 0.04, 0.06]}
        probs = proportions[region]     # obtain population probability for this region
        choice = []
        val = np.random.uniform(0, 1, n)
        for i in range(n):
            if val[i] <= probs[0]:
                choice.append("MS")
            elif val[i] <= probs[1] + probs[0]:
                choice.append("Giant")
            elif val[i] <= probs[2] + probs[1] + probs[0]:
                choice.append("SupGiant")
            else:
                choice.append("WDwarf")
        stars = [Star(region, species) for species in choice]
        return stars
            
    def generate_HR(self, isoradii=False):
        '''Plots a Colour-Magnitude (HR) diagram for this galaxy.     
        Parameters
        ----------
        isoradii : bool
            whether or not to plot constant radius lines on top of the HR diagram
        Returns
        -------
        matplotlib axes object
            The HR diagram. 
        '''
        fig, ax = plt.subplots()
        lumins = [star.get_star_lumin() for star in self.stars]
        temps = [star.get_star_temp() for star in self.stars]
        colours = self.starpositions[3]
        ax.scatter(temps, lumins, color=colours, s=0.2)
        ax.set_facecolor('k'); ax.invert_xaxis(); 
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_xlabel("Temperature (K)"); ax.set_ylabel(r"Solar Luminosities ($L / L_\odot$)")
        
        if isoradii == True:
            textcolour = [0.7, 0.7, 0.7]
            solarradius = 696340000     #initialise variables
            solarlum = 3.828 * 10**26
            sigma = 5.67037 * 10**-8
            xmin, xmax = ax.get_xlim(); ymin, ymax = ax.get_ylim()  #get the current figure bounds so that we don't alter it
            x = np.linspace(xmin, xmax, 2)
            for power in np.arange(-3, 5):
                y = (4 * np.pi * (solarradius * 10.0**power)**2 * sigma * x**4) / solarlum
                ax.plot(x, y, linewidth=0.6, linestyle='--', color=textcolour)
                if power == 0:
                    text = "$R_\odot$"
                elif power == 1:
                    text = "$10R_\odot$"
                else:
                    text = f"$10^{{{power}}} R_\odot$"
                if ymin < max(y) < ymax:    #this makes sure that text doesn't show up outside of the plot bounds
                    ax.text(max(x), max(y), text, color=textcolour, rotation=-23, fontsize=8)
            ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)    #make sure the figure bounds dont change from before
                
    
    def plot_2d(self, fig, ax, spikes=False):
        '''Plots the Galaxy onto predefined matplotlib axes in terms of its equatorial and polar angles. 
        
        Parameters
        ----------
        fig : matplotlib.figure
        ax : matplotlib.axes 
            A predefined matplotlib axes that has been defined by "fig, ax = plt.subplots()"
        spikes : bool
            Whether to show diffraction spikes for bright stars.
        
        Returns
        -------
        No returns, but adds the current Galaxy instance to the matplotlib axes. 
        '''
        x, y, z, colours, scales = self.starpositions
        equat, polar, radius = self.cartesian_to_spherical(x, y, z)
        
        
        scales = scales / (0.05 * radius)
        if spikes == True:
            brightequat, brightpolar, brightscale, brightcolour = [], [], [], np.empty((0, 3))
            for i, scale in enumerate(scales):
                if scale > 2.5:
                    brightequat += [equat[i]]
                    brightpolar += [polar[i]]
                    brightscale = brightscale + [scale / 4]
                    brightcolour = np.append(brightcolour, [colours[i]], axis=0)
            ax.errorbar(brightequat, brightpolar, yerr=brightscale, xerr=brightscale, ecolor=brightcolour, fmt='none', elinewidth=0.3)
        scales = [4 if scale > 4 else scale for scale in scales]
        ax.scatter(equat, polar, s=scales, c=colours)
        ax.set_xlim(0, 360); ax.set_ylim(0, 180)
        ax.set_facecolor('k')
        ax.set_aspect(1)
        fig.tight_layout()
        ax.invert_yaxis()
        ax.set_xlabel("Equatorial Angle (degrees)")
        ax.set_ylabel("Polar Angle (degrees)")
    
    def plot_3d(self, ax, camera=False):
        '''Plots the Galaxy onto predefined 3D matplotlib axes. 
        
        Parameters
        ----------
        ax : matplotlib.axes 
            A predefined matplotlib axes that has been defined by "ax = fig.add_subplot(projection='3d')", 
            where fig is defined by "fig = plt.figure()"
        camera : bool
            whether or not to show a little green pyramid at the origin (0, 0, 0) showing the direction of the camera in the 2d plot
        
        Returns
        -------
        No returns, but adds the current Galaxy instance to the matplotlib axes. 
        '''
        x, y, z, colours, scales = self.starpositions
        ax.scatter(-x, -y, -z, s=scales, c=colours) #need to plot the flipped coordinates for some reason? need to do this to match up with the 2d plot.
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.grid(False)
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        
        # # uncomment the lines below if you want to calibrate galaxy radii, etc
        # ax.view_init(elev=90, azim=0)
        # ax.set_facecolor([0.4, 0.4, 0.4])
        
        # uncomment the lines below for a pretty view
        ax.set_facecolor('k')
        
        if camera == True:
            ax.scatter(0, 0, 0, c='g', s=60, alpha=0.9) #plots the main camera part
            equat = np.array([-30, -30, 30, 30]) + 180
            polar = np.array([-30, 30, 30, -30]) + 90
            distance = self.radius / 2
            equat = np.radians(equat); polar = np.radians(polar)
            x = distance * np.cos(equat) * np.sin(polar)
            y = distance * np.sin(equat) * np.sin(polar)
            z = distance * np.cos(polar)
            ax.scatter(x, y, z, c='g', s=40, alpha=0.9)
            x, y, z = np.append(x, x[0]), np.append(y, y[0]), np.append(z, z[0])
            ax.plot(x, y, z, c='g', linewidth=1)
            for i in range(4):
                ax.plot([0, x[i]], [0, y[i]], [0, z[i]], c='g', linewidth=1)
    
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
    # galaxy = Galaxy('SBb', (40,10,20), 1000, 100, cartesian=True)
    galaxy = Galaxy('Sc', (270, 90, 70), 1000, 100)
    # galaxy2 = Galaxy('E0', (104, 131, 5000), 1000, 100)
    # galaxy3 = Galaxy('Sc', (110, 128, 10000), 1000, 100)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    galaxy.plot_3d(ax, camera=True)
    # ax.set_xlim(-15, 15); ax.set_ylim(-15, 15); ax.set_zlim(-15, 15)
    # ax.set_xlim(-10, 10); ax.set_ylim(-10, 10); ax.set_zlim(-10, 10)
    
    # galaxy.generate_HR(isoradii=True)
    fig, ax = plt.subplots()
    galaxy.plot_2d(fig, ax, spikes=True)
    # galaxy2.plot_2d(fig, ax)
    # galaxy3.plot_2d(fig, ax)

    
if __name__ == "__main__":
    main()