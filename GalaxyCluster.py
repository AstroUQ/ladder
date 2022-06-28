# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 08:38:20 2022

@author: ryanw
"""
import numpy as np
from multiprocessing import Pool
from Galaxy import Galaxy

class GalaxyCluster(object):
    def __init__(self, position, population, cartesian=False, local=False, darkmatter=True, complexity="Normal"):
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
        self.radius = 200 * population**(5/6)
        self.darkmatter = darkmatter
        
        if cartesian:
            self.cartesian = position
            self.spherical = self.cartesian_to_spherical(position[0], position[1], position[2])
        else:
            self.spherical = position
            self.cartesian = self.spherical_to_cartesian(position[0], position[1], position[2])
        
        self.galaxies, self.galaxmasses, self.galaxorbits = self.generate_galaxies(population)
    
    def generate_galaxy(self, species, position):
        ''' Generate a Galaxy class object.
        Parameters
        ----------
        species : str
            The type of galaxy to generate
        position : 3-tuple
            The xyz position of the galaxy (cartesian coords)
        Returns
        -------
        Galaxy : Galaxy object
        '''
        return Galaxy(species, position, cartesian=True)
    
    def generate_galaxies(self, population):
        ''' Uniformly distributes and generates galaxies within a sphere, with a central elliptical galaxy if the cluster
        population is high enough (>=5).
        Parameters
        ----------
        population : int
            The number of galaxies to generate
        Returns
        -------
        galaxies : list
            list of Galaxy class objects that comprise the cluster
        galaxmasses : numpy array
            The mass of each galaxy in the cluster (including dark matter), in units of solar masses.
        orbitradii : numpy array
            The orbital radius of each galaxy from the center of the cluster, in units of parsec
        '''
        theta = np.random.uniform(0, 2*np.pi, population)
        phi = np.random.uniform(-1, 1, population)
        phi = np.arccos(phi)
        
        # dists = np.random.exponential(0.4, population)
        dists = np.random.uniform(0, 1, population)
        R = self.radius * dists**(1/3)
        
        x = R * (np.cos(theta) * np.sin(phi) + np.random.normal(0, 0.1, population))
        y = R * (np.sin(theta) * np.sin(phi) + np.random.normal(0, 0.1, population))
        z = R * (np.cos(phi) + np.random.normal(0, 0.05, population))
        orbitradii = np.sqrt(x**2 + y**2 + z**2)    # orbit radius for each galaxy from the cluster center, in parsecs
        
        # now to move the galaxies to their appropriate position in the sky
        x, y, z = x + self.cartesian[0], y + self.cartesian[1], z + self.cartesian[2]
        
        # determine the types of galaxies in this cluster
        species = self.species_picker(orbitradii, population)
        # print(species)
        if population >= 5:     # need a central elliptical galaxy, so generate n-1 galaxies
            args = [(species[i], [x[i], y[i], z[i]]) for i in range(len(x) - 1)]    # species type at galaxy location
        else:       # no central elliptical needed, so generate n galaxies
            args = [(species[i], [x[i], y[i], z[i]]) for i in range(len(x))]

        if population >= 10:
            args.insert(0, ('cD', self.cartesian))  # insert a cD galaxy in the center of the cluster
        elif population >= 5:
            num = 9 - population    # more populous clusters will have a bigger, more spherical elliptical in their center
            args.insert(0, (f'E{num}', self.cartesian))     # insert an elliptical galaxy in the center of the cluster
            
        # now, use multiprocessing to generate the galaxies in the cluster according to the arguments above and their positions
        with Pool() as pool:
            galaxies = pool.starmap(self.generate_galaxy, args)
        
        galaxmasses = np.zeros(len(galaxies))
        for i, galaxy in enumerate(galaxies):
            galaxmasses[i] = galaxy.galaxymass  # get the mass of each galaxy
            
        return galaxies, galaxmasses, orbitradii
    
    def species_picker(self, orbitradii, population):
        ''' A function to determine the type of galaxies in the cluster, given their orbital radii from the cluster center
        and the cluster population. 
        Parameters
        ----------
        orbitradii : list or numpy array
            The radius of each orbit of the galaxies with the cluster, in units of parsec
        population : int 
            Population of galaxies in the cluster
        Returns
        -------
        types : list
            The species of each galaxy, in the same order as the galaxies in orbitradii
        '''
        if population == 1:
            types = ['S0']
        else:
            types = []
            for i in range(len(orbitradii)):
                prop = orbitradii[i] / self.radius  # determine how far from the center the galaxy is
                elliptcheck = np.random.uniform(0, 1)   # generate a RV to determine whether a galaxy is elliptical
                
                if population >= 5:     # dense cluster
                    # the below makes it more likely for ellipticals the larger the cluster pop is, up to a pop of 10 when it then has 
                    # constant probability
                    elliptical = True if elliptcheck <= (min(0.1 * population, 1) - prop) else False
                    spiral = not elliptical     # of course the galaxy can't be both a spiral and elliptical
                else:
                    elliptical = True if elliptcheck <= 0.1 else False  # about 10% chance of an elliptical galaxy outside of a dense cluster
                    spiral = not elliptical
    
                if spiral == True:
                    barcheck = np.random.uniform(0, 1)
                    barred = True if barcheck <= 0.7 else False     # 70% of spiral galaxies have central bars
                    if barred == True:
                        types.append(np.random.choice(['SBa', 'SBb', 'SBc']))   # choose a barred galaxy
                    else:
                        types.append(np.random.choice(['S0', 'Sa', 'Sb', 'Sc']))    # choose a non-barred galaxy
                else:   # elliptical == True:
                    n = int(7 * min(prop, 1))   # more likely for spherical ellipticals closer to the center of the cluster
                    types.append(f'E{n}')
        return types
    
    def rotation_vels(self):
        ''' Simulates orbit velocities of stars given their distance from the galactic center.
        If the galaxy has dark matter (self.darkmatter == True), then extra mass will be added according to the 
        Navarro-Frenk-White (NFW) dark matter halo mass profile. 
        TO DO: implement different dark matter halo properties for each galaxy type
        Returns
        -------
        np.array:
            2 element numpy array, with each element corresponding to:
                1. vel = the newtonian rotation velocities
                2. darkvel = rotation velocities including dark matter
            if self.darkmatter == False, then darkvel is an array of zeros
        '''
        if self.darkmatter == True:     # time to initialise dark matter properties 
            density = 0.001 # solar masses per cubic parsec
            scalerad = 1.3 * self.radius  # parsec
            Rs = scalerad * 3.086 * 10**16  # convert scalerad to meters
            p0 = density * (1.988 * 10**30 / (3.086 * 10**16)**3) # convert density to kg/m^3
            darkMass = lambda r: p0 / ((r / Rs) * (1 + r / Rs)**2) * (4 / 3 * np.pi * r**3)   # NFW dark matter profile (density * volume)
            
        G = 6.67 * 10**-11
        
        masses, orbits = self.galaxmasses, self.galaxorbits
        # now, create an array that stores the mass and orbital radius of each star in the form of [[m1, r1], [m2,r2], ...]
        MassRadii = np.array([[masses[i] * 1.988 * 10**30, orbits[i] * 3.086 * 10**16] for i in range(len(masses))])
        vel = np.zeros(len(MassRadii)); darkvel = np.zeros(len(MassRadii))  # initialise arrays to store velocities in
        for i in range(len(MassRadii)):
            R = MassRadii[i, 1] 
            # now to sum up all of the mass inside the radius R
            M = sum([MassRadii[n, 0] if MassRadii[n, 1] < R else 0 for n in range(len(MassRadii))])
            vel[i] = (np.sqrt(G * M / R) / 1000)    # calculate newtonian approximation of orbital velocity
            if self.darkmatter == True:
                M += darkMass(R)    # add the average mass of dark matter inside the radius R
                darkvel[i] = (np.sqrt(G * M / R) / 1000)    # newtonian approximation, now including dark matter
        
        velarray = np.array([vel, darkvel]) * np.random.normal(1, 0.01, len(vel))
        

        # now to calculate the direction of the velocity to display the radial component to the observer
        x, y, z, _, _ = self.starpositions
        
        # now we need to transform the galaxy back to the origin with no rotation
        x, y, z = x - self.cartesian[0], y - self.cartesian[1], z - self.cartesian[2]
        points = np.array([x, y, z])
        phi = self.rotation
        
        # rotate galaxy in the reverse order and opposite direction as initially
        points = np.dot(self.galaxyrotation(-phi[2], 'z'), points)
        points = np.dot(self.galaxyrotation(-phi[1], 'y'), points)
        points = np.dot(self.galaxyrotation(-phi[0], 'x'), points)
        
        x, y, z = points
        if self.species[0] == "S":  # spiral galaxy! explanation in the comment block below :)
            theta = np.arctan2(y, x)
            direction = np.array([np.sin(theta), -np.cos(theta), np.random.normal(0, 0.05, len(theta))])
        #         _______                +y             +y|  /
        #         \   _  \               |                | /  \ theta
        # galaxy->/  /_\  \      -x  ____|____ +x         |/____] +x
        #         \  \_/   \             |
        #          \_____  /             |
        #                \/              -y
        # taking the arctan of y/x coordinates of stars gives clockwise circular motion about the galactic center
        # the proportion of motion in the [x, y, z] directions can then be calculated by:
        #     x => sin(theta), since we want theta angles between 0 and pi to have positive x-motion
        #     y => -cos(theta), since we want theta angles between -pi/2 and pi/2 to have negative y-motion
        #     z => normal(0, 0.05) since we want there to be negligible, but random z motion
        else:   # elliptical galaxy! explanation in the comment block below
            direction = np.array([np.zeros(len(x)), np.zeros(len(x)), np.zeros(len(x))])
            for i in range(len(x)):
                xprop = np.random.uniform(-1, 1)
                yprop = np.random.uniform(-1, 1)
                while xprop**2 + yprop**2 > 1:
                    yprop = np.random.uniform(-1, 1)
                zprop = np.sqrt(1 - (xprop**2 + yprop**2))  # 1 = x**2 + y**2 + z**2 => z = sqrt(1 - x**2 - y**2)
                direction[0, i] = xprop; direction[1, i] = yprop; direction[2, i] = zprop
        # the squares of the directional velocity components must add up to one: 1 = xprop**2 + yprop**2 + zprop**2
        # so, we can randomly sample xprop and yprop (between -1 and 1 so that the velocity has random xy direction), 
        # making sure that the sum of their squares is not greater than one. Then, we can subtract the sum of their squares from
        # 1 to find the z component. All of this together gives more or less random direction to the stars about the galactic center. 

        direction = np.dot(self.galaxyrotation(phi[0], 'x'), direction)     # rotate the velocity vectors in the same way as before
        direction = np.dot(self.galaxyrotation(phi[1], 'y'), direction)
        direction = np.dot(self.galaxyrotation(phi[2], 'z'), direction)

        x, y, z, _, _ = self.starpositions  # getting the xyz again is cheaper than doing the rotations again
        
        velprops = np.zeros(len(x))
        for i in range(len(direction[0, :])):
            vector = direction[:, i]    # velocity vector "v"
            coord = np.array([x[i], y[i], z[i]])    # distance vector "d"
            velprops[i] = np.dot(vector, coord) / np.sqrt(x[i]**2 + y[i]**2 + z[i]**2)      # dot product: (v dot d) / ||d||
            # the dot product above gets the radial component of the velocity (thank you Ciaran!! - linear algebra is hard)

        VelObsArray = velarray * velprops   # multiply the actual velocities by the line of sight proportion of the velocity magnitude
        return velarray, VelObsArray
            
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
