# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 13:22:58 2022

@author: ryanw
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import colors
import scipy.integrate as integrate

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
        self.bandlumin = self.generate_BandLumin(self.temperature, self.radius)
            
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
    
    def generate_BandLumin(self, temp, radius):
        ''' Equations retrieved from: http://burro.cwru.edu/academics/Astr221/Light/blackbody.html
        The returned band luminosities are at wavelengths: B = 440nm, G = 500nm, R = 700nm
        Each luminosity value has an uncertainty of +/- 1.5% about a true blackbody. 
        Parameters
        ----------
        temp : float
        radius : float
            Radius of the star in units of solar radii. 
        Returns
        -------
        band luminosities : np.array
            [B, G, R] band luminosities in units of J/nm/s <=> W/nm <=> 10^-9 W/m
            (the planck function has units of J/m^2/nm/s <=> W/m^2/nm <=> 10^-9 W/m^3 )
        '''
        c, h, k = 299792458, 6.626 * 10**-34, 1.38 * 10**-23
        blue, green, red = 440 * 10**-9, 500 * 10**-9, 700 * 10**-9
        planck = lambda x: ((2 * h * c**2) / x**5) * (1 / (np.exp(h * c / (x * k * temp)) - 1)) * 10**-9
        bandLum = lambda x: 4 * np.pi**2 * (696540000 * radius)**2 * planck(x)
        return np.array([bandLum(blue), bandLum(green), bandLum(red)]) * np.random.uniform(0.99, 1.01, 3)
    
    def generate_blackbodycurve(self, markers=True, visible=False):
        ''' Produce a graph of this stars' blackbody curve. 
        Parameters
        ----------
        markers : bool
            whether or not to put the [B, G, R] band luminosity markers on the graph
        visible : bool
            whether or not to plot the visible spectrum overlaid onto the curve
        '''
        temp = self.temperature
        radius = self.radius
        c, h, k = 299792458, 6.626 * 10**-34, 1.38 * 10**-23
        x = np.linspace(1 * 10**-9, 10**-6, 1000)   # the domain for the graph. going from ~0 -> 1000nm
        planck = lambda x: ((2 * h * c**2) / x**5) * (1 / (np.exp(h * c / (x * k * temp)) - 1)) * 10**-9
        bandLum = lambda x: 4 * np.pi**2 * (696540000 * radius)**2 * planck(x)
        lumins = bandLum(x)     # generate the blackbody curve
        
        fig, ax = plt.subplots()
        ax.plot(x * 10**9, lumins, c='k')   # plot the blackbody curve of the star
        ax.set_xlabel(r"Wavelength $\lambda$ (nm)")
        ax.set_ylabel(r"Monochromatic Luminosity (W/nm)")
        ax.set_ylim(ymin=0); ax.set_xlim(xmin=0)
        
        if visible == True:     # plot the visible spectrum under the blackbody curve
            spectrum = np.linspace(1, 1000, 1000)
            colourmap = plt.get_cmap('Spectral_r')  # spectral_r is the visible spectrum going from blue -> red 
            normalize = colors.Normalize(vmin=380, vmax=750) # normalize spectral_r to the wavelength of the visible spectrum
            for i in range(len(spectrum) - 1): # iterate over blocks in the domain, and plot the colour for that block
                where = [True if 380 <= x <= 750 else False for x in [spectrum[i], spectrum[i + 1]]]
                ax.fill_between([spectrum[i], spectrum[i + 1]], [lumins[i], lumins[i + 1]], where=where, 
                                color=colourmap(normalize(spectrum[i])), alpha=0.3)
        if markers == True:     # plot markers for each of the luminosity band values given to the user
            ax.scatter(np.array([440, 500, 700]), self.bandlumin, color=['b', 'g', 'r'])
        
    
    def get_star_lumin(self):
        return self.luminosity
    def get_star_mass(self):
        return self.mass
    def get_star_temp(self):
        return self.temperature
    def get_star_radius(self):
        return self.radius
    def get_star_BandLumin(self):
        return self.bandlumin

def HR_diagram(lumins, temps, colours=None):
    '''Plots a Colour-Magnitude (HR) diagram given some luminosity and temperature values. 
    Parameters
    ----------
    lumins : np.array
        Luminosity values for the stars in units of solar luminosities
    temps : np.array
        Temperature values for the stars in Kelvin
    
    Returns
    -------
    matplotlib axes object
        The HR diagram. 
    '''
    fig, ax = plt.subplots()
    if colours == None:
        ax.scatter(temps, lumins, s=0.2)
    else:
        ax.scatter(temps, lumins, color=colours, s=0.2)
    ax.set_facecolor('k'); ax.invert_xaxis(); 
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel("Temperature (K)"); ax.set_ylabel(r"Solar Luminosities ($L / L_\odot$)")

n = 1000
choice = []
for i in range(n):
    val = np.random.uniform(0, 1)
    if val <= 0.8:
        choice.append("MS")
    elif val <= 0.9:
        choice.append("Giant")
    elif val <= 0.97:
        choice.append("SupGiant")
    else:
        choice.append("WDwarf")
stars = [Star("bulge", species) for species in choice]
lumins = [star.get_star_lumin() for star in stars]
temps = [star.get_star_temp() for star in stars]
colours = [star.get_star_colour() for star in stars]
masses = [star.get_star_mass() for star in stars]
radii = [star.get_star_radius() for star in stars]
bandlums = [star.get_star_BandLumin() for star in stars]

print(bandlums[0], temps[0], lumins[0] * 3.828 * 10**26)
stars[0].generate_blackbodycurve()
stars[0].generate_blackbodycurve(visible=True)

# HR_diagram(lumins, temps, colours)

# fig, ax = plt.subplots()
# logbins = np.logspace(np.log10(min(lumins)), np.log10(max(lumins)), int(n/10))
# ax.hist(lumins, bins=logbins)
# ax.set_xscale('log'); ax.set_xlabel(r"Solar Luminosities ($L / L_\odot$)")

# fig1, ax1 = plt.subplots()
# ax1.hist(masses, bins=30)
# ax1.set_xlabel(r"Solar Masses ($M / M_\odot$)")

# fig2, ax2 = plt.subplots()
# ax2.hist(temps)




    