# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 13:08:36 2022

@author: ryanw
"""
import numpy as np
from tqdm import tqdm     # this is a progress bar for a for loop
from GalaxyCluster import GalaxyCluster

class Universe(object):
    def __init__(self, radius, hubble, clusters, complexity="Normal", homogeneous=False):
        '''
        Parameters
        ----------
        radius : float
            The radius of the observable universe (in units of parsec)
        hubble : float
            The value of the Hubble Constant (in units of km/s/Mpc) - essentially the recession velocity of galaxies
        clusters : int
            The number of galaxy clusters to generate
        complexity : str
            Whether or not the cluster is composed of normal or 'distant' galaxies
        homogeneous : bool
            Whether the universe is homogeneous or not (approx constant density with distance)
        '''
        self.radius = radius
        self.hubble = hubble
        self.clusterpop = clusters
        self.complexity = complexity
        self.homogeneous = homogeneous
        self.clusters, self.clustervels = self.generate_clusters()
        self.galaxies, self.distantgalaxies = self.get_all_galaxies()
        self.supernovae = self.explode_supernovae(min(55, len(self.galaxies) + len(self.distantgalaxies)))
        self.radialvelocities, self.distantradialvelocities = self.get_radial_velocities()
    
    def generate_clusters(self):
        ''' Generate all of the galaxy clusters in the universe.
        Returns
        -------
        clusters : list
            List of GalaxyCluster objects
        '''
        threshold = 100000  # the distance threshold at which galaxies are simulated in low resolution form
        population = self.clusterpop
        clusters = []

        equat = np.random.uniform(0, 360, population)   # generate cluster positions in sky
        # polar = np.random.uniform(0, 180, population)
        polar = np.random.uniform(-1, 1, population)
        polar = np.arccos(polar); polar = np.rad2deg(polar)
        
        lowerbound = 2000 / self.radius     # we want a certain area around the origin to be empty (to make space for the local cluster)
        if self.homogeneous:
            dists = np.random.uniform(lowerbound, 1, population)
        else:
            median = threshold / self.radius    # we want half of the galaxies to be resolved, half to not be
            mean = median / np.log(2)       #  the mean of the exponential distribution is = median / ln(2)
            dists = np.random.exponential(mean**3, population) + lowerbound    # we don't want galaxy clusters within the lowerbounded sphere
        R = self.radius * np.cbrt(dists)
        
        populations = np.random.exponential(8, population)  # generate number of galaxies per cluster
        populations = [1 if pop < 1 else int(pop) for pop in populations]   # make sure each cluster has at least one galaxy
        
        localequat = np.random.uniform(0, 360); localpolar = np.random.uniform(45, 135)     # choose position of local cluster in the sky

        for i in tqdm(range(self.clusterpop)):
            pos = (equat[i], polar[i], R[i])
            if i == self.clusterpop - 1:    # make the last cluster in the list the local cluster
                clusters.append(GalaxyCluster((localequat, localpolar, 2000), 15, local=True, complexity=self.complexity))
            elif R[i] > threshold:  # this must be a distant galaxy
                clusters.append(GalaxyCluster(pos, populations[i], complexity="Distant"))
            else:
                clusters.append(GalaxyCluster(pos, populations[i], complexity=self.complexity))
        
        clustervels = (self.hubble * R / (10**6)) * np.random.normal(1, 0.05, len(R))  # the radial velocity of each cluster according to v = HD
        
        return clusters, clustervels
    
    def get_all_galaxies(self):
        '''
        '''
        clusters = [cluster.galaxies for cluster in self.clusters]
        flatgalaxies = [galaxy for cluster in clusters for galaxy in cluster]
        distantgalaxies = []
        for i, galaxy in enumerate(flatgalaxies):
            if galaxy.complexity == "Distant":
                distantgalaxies.append(galaxy)
                flatgalaxies[i] = None
        galaxies = [galaxy for galaxy in flatgalaxies if galaxy != None]
        return galaxies, distantgalaxies
    def get_all_starpositions(self):
        galaxydata = [galaxy.get_stars() for galaxy in self.galaxies]
        x = [galaxy[0] for galaxy in galaxydata]; x = np.array([coord for xs in x for coord in xs])
        y = [galaxy[1] for galaxy in galaxydata]; y = np.array([coord for xs in y for coord in xs])
        z = [galaxy[2] for galaxy in galaxydata]; z = np.array([coord for xs in z for coord in xs])
        colours = [galaxy[3] for galaxy in galaxydata]; colours = [coord for xs in colours for coord in xs]
        scales = [galaxy[4] for galaxy in galaxydata]; scales = np.array([coord for xs in scales for coord in xs])
        stars = [x, y, z, colours, scales]
        return stars
    def get_blackholes(self):
        blackholes = [galaxy.blackhole for galaxy in self.galaxies]
        distantblackholes = [galaxy.blackhole for galaxy in self.distantgalaxies]
        allblackholes = blackholes + distantblackholes
        return allblackholes
    
    def get_radial_velocities(self):
        stars = self.get_all_starpositions()
        x, y, z, _, _ = stars[0], stars[1], stars[2], stars[3], stars[4]
        _, _, radius = self.cartesian_to_spherical(x, y, z)
        
        locgalaxymovement = self.clusters[-1].directions[:, -1]  # the local galaxy is the last galaxy in the last cluster
        localgalaxydist = self.clusters[-1].galaxies[-1].spherical[2]
        localgalaxy = self.clusters[-1].galaxies[-1]
        closestar = min(localgalaxy.starorbits, key=lambda x:abs(x - localgalaxydist))
        closestarindex = np.where(localgalaxy.starorbits == closestar)
        approxlocalstarvel = localgalaxy.starvels[1, closestarindex[0]]
        localstarmovement = approxlocalstarvel * np.array([0, 1, np.random.normal(0, 0.05)])
        
        galaxydirection = []
        distantgalaxydirection = []
        for cluster in self.clusters:
            for i in range(len(cluster.galaxies)):
                # add the vector of the local galaxy movement with the current galaxy movement
                if cluster.galaxies[i].complexity != "Distant":
                    galaxydirection.append(cluster.directions[:, i]) 
                else:
                    distantgalaxydirection.append(cluster.directions[:, i]) 
        galaxydirection = np.array(galaxydirection)
        distantgalaxydirection = np.array(distantgalaxydirection)
        DGcartesians = np.array([galaxy.cartesian for galaxy in self.distantgalaxies])    # distant galaxy cartesians
        DGx, DGy, DGz = DGcartesians[:, 0], DGcartesians[:, 1], DGcartesians[:, 2]     # distant galaxy x, distant galaxy y, etc
        DGradius = [galaxy.spherical[2] for galaxy in self.distantgalaxies]
        
        stardirections = []
        distantgalaxyvectors = []
        k = 0; m = 0
        for h, cluster in enumerate(self.clusters):
            galaxyvels = cluster.galaxvels[1, :]
            for i, galaxy in enumerate(cluster.galaxies):
                if galaxy.complexity != "Distant":  # close galaxy, so we're dealing with stars
                    stardirection = galaxy.directions
                    starvels = galaxy.starvels[1, :]
                    galaxyvel = galaxyvels[i]
                    for j in range(len(stardirection[0, :])):
                        if h == len(self.clusters) - 1 and i == len(cluster.galaxies) - 1:      # must be the local galaxy
                            stardirection[:, j] = (stardirection[:, j] * starvels[j])
                        else:
                            stardirection[:, j] = (stardirection[:, j] * starvels[j]) + (galaxydirection[k] * galaxyvel)
                        stardirections.append(stardirection[:, j])
                    k += 1
                else:   # distant galaxy, so we're dealing with galaxy as a whole
                    vector = distantgalaxydirection[m] * galaxyvels[i]
                    distantgalaxyvectors.append(vector)
                    m += 1
        k = 0
        m = 0
        obsvel = np.zeros(len(stardirections))
        distantobsvel = np.zeros(len(distantgalaxyvectors))
        for h, cluster in enumerate(self.clusters):
            if h != len(self.clusters) - 1:
                clustervel = self.clustervels[h]
                addclustervel = True
            else:
                addclustervel = False
            for i, galaxy in enumerate(cluster.galaxies):
                if galaxy.complexity != "Distant":  # close galaxy, so we're working with individual stars
                    if addclustervel == False and i == len(cluster.galaxies) - 1:
                        addgalaxyvel = False
                    else:
                        addgalaxyvel = True
                    for j in range(len(galaxy.stars)):
                        if addgalaxyvel:
                            vector = stardirections[k] + locgalaxymovement + localstarmovement    # velocity vector "v"
                        else:
                            vector = stardirections[k] + localstarmovement
                        coord = np.array([x[k], y[k], z[k]])    # distance vector "d"
                        obsvel[k] = np.dot(vector, coord) / radius[k]      # dot product: (v dot d) / ||d||
                        # the dot product above gets the radial component of the velocity (thank you Ciaran!! - linear algebra is hard)
                        if addclustervel:
                            obsvel[k] += clustervel
                        k += 1
                else:   # distant galaxy, so we're working with galaxies as a whole
                    vector = distantgalaxyvectors[m] + locgalaxymovement + localstarmovement
                    coord = np.array([DGx[m], DGy[m], DGz[m]])
                    distantobsvel[m] = np.dot(vector, coord) / DGradius[m]
                    if addclustervel:
                        distantobsvel[m] += clustervel
                    m += 1
        return obsvel, distantobsvel
    
    def explode_supernovae(self, frequency):
        '''
        Parameters
        ----------
        frequency : int
            The number of supernovae to generate
        '''
        allgalaxies = self.distantgalaxies + self.galaxies
        indexes = np.random.uniform(0, len(allgalaxies) - 1, frequency - 2)
        closeindexes = len(allgalaxies) - np.random.uniform(1, 14, 2)     # gets two indexes within the last 14 of the galaxy list
        indexes = np.append(indexes, closeindexes); np.random.shuffle(indexes)
        galaxies = [allgalaxies[int(i)] for i in indexes]
        positions = np.array([galaxy.spherical for galaxy in galaxies])
        # intrinsic = 1.5 * 10**44 / (4 * np.pi * (7 * 10**6)**2)     # rough energy release of R=7000km white dwarf Type Ia supernova (W/m^2)
        peakmag = -18.4; sunmag = 4.74; sunlumin = 3.828 * 10**26   # peak magnitude of a type 1a supernova (M_V), bol abs mag of the sun, bol lumin of the sun
        peaklumin = sunlumin * 10**((peakmag - sunmag) / (-2.5))    # this is the mag/lumin formula rearranged to give L
        intrinsicflux = peaklumin / (4 * np.pi * (5 * 10**6)**2)    # rough energy release of R=7000km white dwarf Type Ia supernova (W/m^2)
        intrinsicflux *= (10 * 3.086 * 10**16)**2                   # account for the peakmag being at 10pc
        distances = positions[:, 2] * 3.086 * 10**16    # convert from parsec to meters
        peakfluxes = (intrinsicflux / distances**2) * np.random.normal(1, 0.01, frequency)
        skypositions = [positions[:, 0] + np.random.normal(0, 0.01, frequency), 
                        positions[:, 1] + np.random.normal(0, 0.01, frequency)]   # [equat, polar]
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
            