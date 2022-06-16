# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 09:52:26 2022

@author: ryanw

"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy
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
        return 4 * np.pi * R**2 * sigma * temps**4 * mult * np.random.normal(1, 0.05)

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
        '''Approximations were retrieved from https://en.wikipedia.org/wiki/Planckian_locus
        
        Alternate::
            # Mitchell Charity <mcharity@lcs.mit.edu>
            # http://www.vendian.org/mncharity/dir3/blackbody/
            # Version 2001-Jun-22
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
        scale = 3 * np.log(self.luminosity + 1)
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
        self.species = species
        
        self.population = population
        self.radius = radius
        
        if cartesian:
            self.cartesian = position
            self.spherical = self.cartesian_to_spherical(position)
        else:
            self.spherical = position
            self.cartesian = self.spherical_to_cartesian(position[0], position[1], position[2])
            
        self.starpositions = self.generate_galaxy()
        
        
    
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
    
    def get_params(self):
        '''
        Returns
        -------
        params : np.array
            if spiral:
                [bulgepop, diskpop, spiralpop, lowerspiral, upperspiral, spiralyoung, spiralold, ]
            if elliptical:
                []
        '''
        speciesparams = {"Sa" : 1,
                         "Sb" : 1,
                         "Sc" : 1,
                         "SBa" : 1,
                         "SBb" : 1,
                         "SBc" : 1,
                         "S0" : 1,
                         "E" : 1}
        params = speciesparams[self.species]
        return params
    
    def generate_spiral(self, population, radius):
        '''Barred spiral galaxies generated using Fermat's spiral, and standard spiral galaxies generated using Archimedean spiral. 
        '''
        speciesindex = {"S0":0, "Sa":1, "Sb":2, "Sc":3, "SBa":4, "SBb": 5, "SBc":6}
        radii = [None, 10, 6, 2.1, 4.2, 3.7, 3.7]                                                #radius divisors 
        wrap = [[None, None], [0.9, 4 * np.pi], [0.7, 2 * np.pi], [0.2, 0.8 * np.pi], [np.pi / 2.1, 3 * np.pi], [np.pi / 2.1, 2 * np.pi], [np.pi / 2.1, 1.15 * np.pi]]   #angle extents EXPERIMENTAL
        
        mult, spiralwrap = [param[speciesindex[self.species]] for param in [radii, wrap]]
        
        if mult != None:
            diskpop = int(0.5 * population); bulgepop = int(0.2 * population); spiralpop = int(0.3 * population)
        else: #S0 Galaxy
            diskpop = int(0.8 * population); bulgepop = int(0.2 * population);
        bulgeradius = radius / 10
        
        diskdists = np.random.exponential(radius / 4, size = diskpop)
        
        theta = np.random.uniform(0, 2*np.pi, diskpop)
        
        #this defines the disk star positions
        diskx = np.cos(theta[:diskpop]) * diskdists
        disky = np.sin(theta[:diskpop]) * diskdists
        diskz = np.zeros(diskpop) + 0.02 * radius * np.random.randn(diskpop)
        diskstars = self.generate_stars("disk", diskpop)
        disktemps = [star.get_star_temp() for star in diskstars]
        diskcolours = [star.get_star_colour() for star in diskstars]
        diskscales = [star.get_star_scale() for star in diskstars]
        
        #this defines the bulge star positions
        if self.species[:2] == "SB":
            barpop = int(bulgepop / 2); bulgepop = int(bulgepop / 2)
            bulgedists = np.random.weibull(1.5 * bulgeradius, size = bulgepop) * np.random.normal(1, 0.05, bulgepop)
            theta = np.random.uniform(0, 2*np.pi, bulgepop)
            bulgex = np.cos(theta) * bulgedists
            bulgey = np.sin(theta) * bulgedists
            bulgez = np.random.normal(0, 1.4/3 * bulgeradius, bulgepop)
            bulgestars = self.generate_stars("bulge", bulgepop)
            bulgetemps = [star.get_star_temp() for star in bulgestars]
            bulgecolours = [star.get_star_colour() for star in bulgestars]
            bulgescales = [star.get_star_scale() for star in bulgestars]
            
            barradius = 0.3 * radius
            barx = np.random.normal(0, 0.07 * barradius, barpop)
            bary = barradius * (np.geomspace(0.3, 1.1, barpop) * np.random.choice([-1, 1], barpop) + np.random.normal(0, 0.1, barpop))
            barz = np.random.normal(0, 0.05 * barradius, barpop)
            barstars = self.generate_stars("bulge", barpop)
            bartemps = [star.get_star_temp() for star in barstars]
            barcolours = [star.get_star_colour() for star in barstars]
            barscales = [star.get_star_scale() for star in barstars]
            bulgex = np.append(bulgex, barx); bulgey = np.append(bulgey, bary); bulgez = np.append(bulgez, barz)
            bulgecolours = np.append(bulgecolours, barcolours, axis=0)
            bulgescales = np.append(bulgescales, barscales, axis=0)
        else:
            bulgedists = np.random.weibull(1.5 * bulgeradius, size = bulgepop) * np.random.normal(1, 0.05, bulgepop)
            theta = np.random.uniform(0, 2*np.pi, bulgepop)
            bulgex = np.cos(theta) * bulgedists
            bulgey = np.sin(theta) * bulgedists
            bulgez = np.random.normal(0, 1.4/3 * bulgeradius, bulgepop)
            bulgestars = self.generate_stars("bulge", bulgepop)
            bulgetemps = [star.get_star_temp() for star in bulgestars]
            bulgecolours = [star.get_star_colour() for star in bulgestars]
            bulgescales = [star.get_star_scale() for star in bulgestars]
        
        
        spiralx, spiraly, spiralz, spiralcolours, spiralscales = [], [], [], np.empty((0,3)), []
        
        #the following is experimental, and models barred spirals accurately
        if mult != None and self.species[:2] == "SB":       #barred spiral
            lower, upper = spiralwrap
            youngpop, oldpop =  int(spiralpop / 2), int(spiralpop / 2)
            youngstars = ["ys", youngpop, 0, 0.04, 0.01, 0.005, 10000, 6000] #[pop, lag, scatter, scatter2, zscatter, tempmean, tempshift]
            oldstars = ["os", oldpop, 0.2, 0.08, 0.015, 0.01, 4000, 1000]
            spiralstars = [youngstars, oldstars]

            for [region, pop, lag, scatter, scatter2, zscatter, tempmean, tempshift] in spiralstars:
                if speciesindex[self.species] >= 5:
                    spiralangle = np.geomspace(lower, upper, pop)
                else:
                    spiralangle = np.linspace(lower, upper, pop)
                reflect = np.random.choice([-1, 1], pop)
                x = (radius / mult) * (spiralangle**(1/2) * np.cos(spiralangle + lag)  * np.random.normal(1, scatter, pop) * reflect) # + np.random.normal(0, scatter2, pop))
                y = (radius / mult) * (spiralangle**(1/2) * np.sin(spiralangle + lag) * np.random.normal(1, scatter, pop) * - reflect) # + np.random.normal(0, scatter2, pop))
                z = np.random.normal(0, zscatter * radius, pop)
                stars = self.generate_stars(region, pop)
                temps = [star.get_star_temp() for star in stars]
                colours = [star.get_star_colour() for star in stars]
                scales = [star.get_star_scale() for star in stars]
                spiralx = np.append(spiralx, x); spiraly = np.append(spiraly, y); spiralz = np.append(spiralz, z)
                spiralcolours = np.append(spiralcolours, colours, axis=0)
                spiralscales = np.append(spiralscales, scales, axis=0)
        elif mult != None:          #standard spiral 
            lower, upper = spiralwrap
            youngpop, oldpop = int(spiralpop / 2), int(spiralpop / 2)
            youngstars = ["ys", youngpop, 0, 0.04, 0.01, 0.005, 10000, 6000] #[pop, lag, scatter, scatter2, zscatter, tempmean, tempshift]
            oldstars = ["os", oldpop, 0.2, 0.08, 0.015, 0.01, 4000, 1000]
            spiralstars = [youngstars, oldstars]
            
            for [region, pop, lag, scatter, scatter2, zscatter, tempmean, tempshift] in spiralstars:
                if speciesindex[self.species] >= 5:
                    spiralangle = np.geomspace(lower, upper, pop)
                else:
                    spiralangle = np.linspace(lower, upper, pop)
                reflect = np.random.choice([-1, 1], pop)
                x = (radius / mult) * (spiralangle * np.cos(spiralangle + lag)  * np.random.normal(1, scatter, pop) * reflect + np.random.normal(0, scatter2, pop))
                y = (radius / mult) * (spiralangle * np.sin(spiralangle + lag) * np.random.normal(1, scatter, pop) * - reflect + np.random.normal(0, scatter2, pop))
                z = np.random.normal(0, zscatter * radius, pop)
                stars = self.generate_stars(region, pop)
                temps = [star.get_star_temp() for star in stars]
                colours = [star.get_star_colour() for star in stars]
                scales = [star.get_star_scale() for star in stars]
                spiralx = np.append(spiralx, x); spiraly = np.append(spiraly, y); spiralz = np.append(spiralz, z)
                spiralcolours = np.append(spiralcolours, colours, axis=0)
                spiralscales = np.append(spiralscales, scales, axis=0)
                
        x = np.append(diskx, np.append(bulgex, spiralx)); y = np.append(disky, np.append(bulgey, spiraly)); z = np.append(diskz, np.append(bulgez, spiralz))
        colours = np.append(diskcolours, np.append(bulgecolours, spiralcolours, axis=0), axis=0)
        scales = np.append(diskscales, np.append(bulgescales, spiralscales, axis=0), axis=0)
        return x, y, z, colours, scales
    
    def generate_elliptical(self, population, radius):
        '''TODO: make more elliptical spirals. Maybe multiply by the reciprocal of the distance from the center in x,y coords?
        '''
        centralpop = int(0.2 * population); spherepop = int(0.8 * population)
        
        centralradius = radius / 6
        
        centraldists = np.random.exponential(radius / 6, size = centralpop)
        # spheredists = np.random.weibull(radius, size = spherepop) * np.random.normal(1, 0.05, spherepop)
        spheredists = np.random.uniform(radius/20, radius, spherepop) * np.random.normal(1, 0.05, spherepop)
        
        theta = np.random.uniform(0, 2*np.pi, population)
        phi = np.random.uniform(-1, 1, population)
        phi = np.arccos(phi)
        spheredists = np.random.exponential(0.07 * radius, spherepop)#np.random.uniform(0, 1, spherepop)
        centraldists = np.random.exponential(centralradius, centralpop)#np.random.uniform(0, 1, centralpop)
        centralR = centralradius * centraldists**(1/3)
        sphereR = radius * spheredists**(1/3)
        
        centralx = centralR * (np.cos(theta[:centralpop]) * np.sin(phi[:centralpop]) + np.random.normal(0, 0.1, centralpop))
        centraly = centralR * (np.sin(theta[:centralpop]) * np.sin(phi[:centralpop]) + np.random.normal(0, 0.1, centralpop))
        centralz = centralR * (np.cos(phi[:centralpop]) + np.random.normal(0, 0.1, centralpop))
        
        centralstars = self.generate_stars("bulge", centralpop)
        centraltemps = [star.get_star_temp() for star in centralstars]
        centralcolours = [star.get_star_colour() for star in centralstars]
        centralscales = [star.get_star_scale() for star in centralstars]
        
        spherex = sphereR * (np.cos(theta[centralpop:]) * np.sin(phi[centralpop:]) + np.random.normal(0, 0.1, spherepop))
        spherey = sphereR * (np.sin(theta[centralpop:]) * np.sin(phi[centralpop:]) + np.random.normal(0, 0.1, spherepop))
        spherez = sphereR * (np.cos(phi[centralpop:])  + np.random.normal(0, 0.1, spherepop))
        
        spherestars = self.generate_stars("disk", spherepop)
        spheretemps = [star.get_star_temp() for star in spherestars]
        spherecolours = [star.get_star_colour() for star in spherestars]
        spherescales = [star.get_star_scale() for star in spherestars]
        
        x = np.append(centralx, spherex); y = np.append(centraly, spherey); z = np.append(centralz, spherez)
        colours = np.append(centralcolours, spherecolours, axis=0)
        scales = np.append(centralscales, spherescales, axis=0)
        return x, y, z, colours, scales
    
    def generate_galaxy(self):
        '''Generate random stars according to species type of galaxy. 
        
        Returns
        -------
        numpy array (x4)
            Cartesian coordinates [x, y, z] of each of the stars in this galaxy, as well as an array of colours for each star. 
        '''
        population, radius = self.population, self.radius
        
        if self.species[0] == 'E':  #elliptical galaxy
            x, y, z, colours, scales = self.generate_elliptical(population, radius)
        else:                       #spiral galaxy
            x, y, z, colours, scales = self.generate_spiral(population, radius)
        
        points = np.array([x, y, z])
        phi = np.random.uniform(0, 2*np.pi, 3)
        
        #rotate the galaxy randomly
        points = np.dot(self.galaxyrotation(phi[0], 'x'), points)
        points = np.dot(self.galaxyrotation(phi[1], 'y'), points)
        points = np.dot(self.galaxyrotation(phi[2], 'z'), points)
        x0, y0, distance = self.cartesian
        x, y, z = points[0] + x0, points[1] + y0, points[2] + distance  #move the galaxy away from the origin to its desired position
        return x, y, z, colours, scales
        
    def get_stars(self):
        return self.starpositions
    
    def generate_stars(self, region, n):
        '''
        '''
        proportions = {"ys":[0.82, 0.1, 0.07, 0.01], "os":[0.75, 0.15, 0.05, 0.05],
                       "disk":[0.9, 0.05, 0.02, 0.03], "bulge":[0.8, 0.1, 0.04, 0.06]}
        probs = proportions[region]
        choice = []
        val = np.random.uniform(0, 1, n)
        for i in range(n):
            if val[i] <= probs[0]:
                choice.append("MS")
            elif val[i] <= probs[1]:
                choice.append("Giant")
            elif val[i] <= probs[2]:
                choice.append("SupGiant")
            else:
                choice.append("WDwarf")
        stars = [Star(region, species) for species in choice]
        return stars
            
    
    def plot_2d(self, fig, ax):
        '''Plots the Galaxy onto predefined matplotlib axes in terms of its equatorial and polar angles. 
        
        Parameters
        ----------
        ax : matplotlib.axes 
            A predefined matplotlib axes that has been defined by "fig, ax = plt.subplots()"
        
        Returns
        -------
        No returns, but adds the current Galaxy instance to the matplotlib axes. 
        '''
        x, y, z, colours, scales = self.starpositions
        equat, polar, radius = self.cartesian_to_spherical(x, y, z)
        
        # scales = [colour[2] for colour in colours] / (0.1 * radius)
        ax.scatter(equat, polar, s=scales, c=colours)
        ax.set_xlim(0, 360); ax.set_ylim(0, 180)
        ax.set_facecolor('k')
        ax.set_aspect(1)
        fig.tight_layout()
        ax.invert_yaxis()
    
    def plot_3d(self, ax):
        '''Plots the Galaxy onto predefined 3D matplotlib axes. 
        
        Parameters
        ----------
        ax : matplotlib.axes 
            A predefined matplotlib axes that has been defined by "ax = fig.add_subplot(projection='3d')", 
            where fig is defined by "fig = plt.figure()"
        
        Returns
        -------
        No returns, but adds the current Galaxy instance to the matplotlib axes. 
        '''
        equat, polar, dist, colours, scales = self.starpositions
        
        
        # scales = [colour[2] for colour in colours]
        ax.scatter(equat, polar, dist, s=scales, c=colours)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.grid(False)
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.set_facecolor('k')
    
    def cartesian_to_spherical(self, x, y, z):
        '''
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
        equat = np.zeros(len(x))
        for i, xcoord in enumerate(x):
            if xcoord == 0:
                equat[i] = np.sign(y[i]) * np.pi / 2
            elif xcoord < 0:
                if y[i] >= 0:
                    equat[i] = np.arctan(y[i] / xcoord) + np.pi
                else:
                    equat[i] = np.arctan(y[i] / xcoord) - np.pi
            else:
                equat[i] = np.arctan(y[i] / xcoord)
        polar = np.arccos(z / radius)
        polar = 180 / np.pi * polar; equat = 180 / np.pi * equat
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
        equat, polar = [np.pi / 180 * angle for angle in [equat, polar]]
        x = distance * np.cos(equat) * np.sin(polar)
        y = distance * np.sin(equat) * np.sin(polar)
        z = distance * np.cos(polar)
        return (x, y, z)
    
    

def main():
    galaxy = Galaxy('SBb', (100,130,100), 1000, 10)
    # galaxy2 = Galaxy('E0', (104, 131, 500), 1000, 10)
    # galaxy3 = Galaxy('Sc', (110, 128, 1000), 1000, 10)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    galaxy.plot_3d(ax)
    # ax.set_xlim(-15, 15); ax.set_ylim(-15, 15); ax.set_zlim(-15, 15)
    # fig, ax = plt.subplots()
    # galaxy.plot_2d(fig, ax)
    # galaxy2.plot_2d(fig, ax)
    # galaxy3.plot_2d(fig, ax)

    
if __name__ == "__main__":
    main()