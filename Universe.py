# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 13:08:36 2022

@author: ryanw
"""
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from tqdm import tqdm     # this is a progress bar for a for loop
from GalaxyCluster import GalaxyCluster

class Universe(object):
    def __init__(self, radius, clusters, hubble=None, blackholes=True, darkmatter=True, complexity="Comprehensive", 
                 variablestars=True, homogeneous=False):
        ''' Generate a universe consisting of Star, BlackHole, Galaxy, and GalaxyCluster objects. 
        Parameters
        ----------
        radius : float
            The radius of the observable universe (in units of parsec)
        hubble : float
            The value of the Hubble Constant (in units of km/s/Mpc) - essentially the recession velocity of galaxies
        clusters : int
            The number of galaxy clusters to generate
        complexity : str
            One of {"Comprehensive", "Normal", "Basic"} which chooses how complicated the data analysis will be. 
            Comprehensive - some galaxies will and won't have darkmatter/blackholes, and have many stars.
            Normal - all galaxies either will or won't have darkmatter/blackholes, and have many stars.
            Basic - all galaxies either will or won't have darkmatter/blackholes, and have few stars.
        homogeneous : bool
            Whether the universe is homogeneous or not (approx constant density with distance)
        '''
        self.radius = radius
        self.clusterpop = clusters
        self.hubble = self.choosehubble() if hubble == None else hubble
        self.blackholes = blackholes 
        self.darkmatter = darkmatter
        self.complexity = complexity
        self.variablestars = self.determine_variablestars(variablestars)
        self.homogeneous = homogeneous
        self.clusters, self.clustervels, self.clusterdists = self.generate_clusters()
        self.galaxies, self.distantgalaxies = self.get_all_galaxies()
        self.supernovae = self.explode_supernovae(min(55, len(self.galaxies) + len(self.distantgalaxies)))
        self.radialvelocities, self.distantradialvelocities = self.get_radial_velocities()
    
    def choosehubble(self):
        ''' Randomly generate a value for hubble's constant, with random sign. 
        Returns
        -------
        float
            The Hubble constant, in units of km/s/Mpc
        '''
        return np.random.choice([-1, 1]) * np.random.uniform(1000, 5000)
    
    def determine_variablestars(self, variable):
        ''' Determine the type of variable stars in the universe, and their period-luminosity relationships.
        Generates 2 or 3 types of variable stars, with periodicity "Short" (18-30 hours), "Long" (38-55 hours), or "Longest"
        (75-100 hours), with "Longest" only appearing in 1/3 datasets. 
        Parameters
        ----------
        variable : bool
            Whether or not this universe should have variable stars
        Returns
        -------
        variablestars : list
            A list composed of [bool, variableparams * n], where the bool corresponds to if this universe has variable stars, 
            variable params : list
                [RoughPeriod, Curvetype, Gradient, Yintercept], where Curvetype is one of {"Saw", "Tri", "Sine"} (Sawtooth wave,
                Triangle wave, and Sine wave respectively) which determines the shape of the variable lightcurve. RoughPeriod, 
                Gradient and Yintercept are all float type, where gradient and y-intercept correspond to the parameters of the 
                period-luminosity relationship trend (period = gradient * log10(luminosity) + yint). Rough period is the roughly
                accurate mean period of the period-lum trend.
            and n is the number of variable types in the universe (2/3 chance of n=2, 1/3 chance of n=3)
        '''
        if variable:
            # first, we want to determine the shape of each variable type lightcurve. Choose randomly from Saw Tri and Sine
            indexes = np.array([0, 1, 2]); np.random.shuffle(indexes)
            curvetypes = ["Saw", "Tri", "Sine"]; curvetypes = [curvetypes[index] for index in indexes]
            
            prob = np.random.uniform(0, 1); types = 2 if prob <= 0.66 else 3    # determine whether to do 2 or 3 variable types
            
            shortperiod = np.random.uniform(18, 30); longperiod = np.random.uniform(38, 55)     # randomly determine period of the variables within some allowed range
            longestperiod = np.random.uniform(75, 100)
            periods = [shortperiod, longperiod, longestperiod]
            signs = np.random.choice([-1, 1], 3)    # randomly choose sign of the period-luminosity trend
            variablestars = [True]
            
            # we want to find the equation of a line given two points (high lumin and low lumin)
            shortlower = 3.5*10**5; shortupper = 5*10**6    # lower and upper luminosity bounds
            shortperiodL = np.random.uniform(0.85, 0.95); shortperiodU = np.random.uniform(1.05, 1.25)     # proportion of lower and upper period bounds
            shortgradient = signs[0] * (shortperiod * (shortperiodU - shortperiodL)) / (np.log10(shortupper / shortlower))    # m = sign * rise/run
            shortyint = (shortperiodL * shortperiod) - (shortgradient * np.log10(shortlower))   # y = mx + c => c = y - mx
            
            # as above, but for the long and longest variable types
            longlower = 100; longupper = 700
            longperiodL = np.random.uniform(0.85, 0.95); longperiodU = np.random.uniform(1.05, 1.25)
            longgradient = signs[1] * (longperiod * (longperiodU - longperiodL)) / np.log10(longupper / longlower)
            longyint = (longperiodL * longperiod) - (longgradient * np.log10(longlower))
            
            longestlower = 100; longestupper = 700
            longestperiodL = np.random.uniform(0.9, 0.98); longestperiodU = np.random.uniform(1.05, 1.25)
            longestgradient = signs[2] * (longestperiod * (longestperiodU - longestperiodL)) / np.log10(longestupper / longestlower)
            longestyint = (longestperiodL * longestperiod) - (longestgradient * np.log10(longestlower))
            
            gradients = [shortgradient, longgradient, longestgradient]
            yints = [shortyint, longyint, longestyint]
            for i in range(types):  # append the period-lum parameters to a list to be sent across the universe
                variablestars.append([periods[i], curvetypes[i], gradients[i], yints[i]])
            return variablestars
        else:
            return [False, [], []]  # if we dont want variable stars, an 'empty' list still needs to be sent out
    
    def generate_clusters(self):
        ''' Generate all of the galaxy clusters in the universe.
        Returns
        -------
        clusters : list
            List of GalaxyCluster objects
        '''
        threshold = 30000  # the distance threshold at which galaxies are simulated in low resolution form
        population = self.clusterpop
        clusters = []

        equat = np.random.uniform(0, 360, population)   # generate cluster positions in sky
        # polar = np.random.uniform(0, 180, population)
        polar = np.random.uniform(-1, 1, population)
        polar = np.arccos(polar); polar = np.rad2deg(polar)
        
        lowerbound = 5000      # we want a certain area around the origin to be empty (to make space for the local cluster)
        if self.homogeneous:
            dists = np.random.uniform(lowerbound / self.radius, 1, population)
            R = self.radius * np.cbrt(dists)
        else:
            # proportion = 1/10    # proportion of total galaxies that you want to be resolved, 
            # # cdf of the exponential distribution is F(x) = 1 - exp(- x / b), where b is the 'scale', or mean, which goes into the numpy exponential function
            # # rearranging this, for some prop 'p', we get b = -x / ln(1 - p), and so
            # mean = - threshold / np.log(1 - proportion)
            # mean = mean / self.radius       # get the mean as a proportion of total radius
            # dists = np.random.exponential(mean, population) + lowerbound / self.radius
            # R = self.radius * dists
                
            ### -- experimental: truncated exponential distribution for galaxy clusters in the universe -- ###
            # in its current state, the clusters are distributed according to two exponential distributions:
            #            |close    |_        distant            the left distribution makes up close clusters, and 
            #            |clusters/| \_      clusters           clusters are more likely close to the threshold
            # frequency  |      _/ |   \__                      the right distribution makes up distant clusters, and 
            #            |  ___/   |      \______               clusters get less probable with distance
            #            |_/_______|_____________\_____
            #                      |    distance (kpc)
            #                      ^threshold (usually 30kpc)
            proportion = 2 / np.sqrt(self.clusterpop)    # proportion of total galaxies that you want to be resolved, 1/sqrt(n) gives a good, scaleable number.
            closepop = int(proportion * self.clusterpop); farpop = int(self.clusterpop - closepop)  # find populations of each category
            closescale = 2/5 * threshold    # the mean of the close distribution will actually be at about 3/5 of the threshold
            # now, define the close distribution using scipy truncated exponential dist. b is a form of the upper bound.
            # loc is the lower bound of the distribution and scale is the mean value after the lowerbound (i think?)
            closedistribution = stats.truncexpon(b=(threshold - lowerbound)/closescale, loc=lowerbound, scale=closescale)
            # now, to get the increasing shape we minus the random variables from the upper bound, and add the lower bound again to account for the shift
            closedists = threshold - closedistribution.rvs(closepop) + lowerbound       # make 'closepop' number of random variables
            # most of the steps below are analogous, but for the distant galaxy clusters
            farscale = self.radius / 2
            fardistribution = stats.truncexpon(b=(self.radius - threshold)/farscale, loc=threshold, scale=farscale)
            fardists = fardistribution.rvs(farpop)
            R = np.append(closedists, fardists)
            
        populations = np.random.exponential(8, population)  # generate number of galaxies per cluster
        populations = [1 if pop < 1 else int(pop) for pop in populations]   # make sure each cluster has at least one galaxy
        
        localequat = np.random.uniform(0, 360); localpolar = np.random.uniform(45, 135)     # choose position of local cluster in the sky

        for i in tqdm(range(self.clusterpop)):
            pos = (equat[i], polar[i], R[i])
            if i == self.clusterpop - 1:    # make the last cluster in the list the local cluster
                clusters.append(GalaxyCluster((localequat, localpolar, 2000), 15, local=True, blackholes=self.blackholes, 
                                              darkmatter=self.darkmatter, complexity=self.complexity, variable=self.variablestars))
            elif R[i] > threshold:  # this must be a distant galaxy
                clusters.append(GalaxyCluster(pos, populations[i], complexity="Distant", variable=self.variablestars))
            else:
                clusters.append(GalaxyCluster(pos, populations[i], blackholes=self.blackholes, darkmatter=self.darkmatter, 
                                              complexity=self.complexity, variable=self.variablestars))
        
        clustervels = (self.hubble * R / (10**6)) * np.random.normal(1, 0.05, len(R))  # the radial velocity of each cluster according to v = HD
        
        return clusters, clustervels, R
    
    def get_all_galaxies(self):
        ''' Obtain flattened lists of all of the galaxies and distant galaxies in the Universe.
        Returns
        -------
        galaxies : list
            List of all the Galaxy objects in the universe
        distantgalaxies : list
            List of all of the Galaxy objects in the universe that have the complexity="Distant" trait.
        '''
        clusters = [cluster.galaxies for cluster in self.clusters]
        flatgalaxies = [galaxy for cluster in clusters for galaxy in cluster]   # flatten the list of galaxies so that there are no nested lists
        distantgalaxies = []    # initialise list to store distant galaxies
        for i, galaxy in enumerate(flatgalaxies):
            if galaxy.complexity == "Distant":
                distantgalaxies.append(galaxy)  # move this galaxy into the distant galaxy list
                flatgalaxies[i] = None      # set the moved galaxy to "None" so that it may be ignored later
        galaxies = [galaxy for galaxy in flatgalaxies if galaxy != None]    # get all of the non-distant galaxies (ignore None)
        return galaxies, distantgalaxies
    
    def get_all_starpositions(self):
        ''' Return a list of all of the stars in the universe in the same format as a "galaxy.starpositions" call, that is
        [x, y, z, colours, scales]
        Returns
        -------
        stars : list
            A 5xn list of all of the stars in the format described in the description.
        '''
        galaxydata = [galaxy.get_stars() for galaxy in self.galaxies]
        x = [galaxy[0] for galaxy in galaxydata]; x = np.array([coord for xs in x for coord in xs])     # flatten the x data and convert to numpy array
        y = [galaxy[1] for galaxy in galaxydata]; y = np.array([coord for xs in y for coord in xs])
        z = [galaxy[2] for galaxy in galaxydata]; z = np.array([coord for xs in z for coord in xs])
        colours = [galaxy[3] for galaxy in galaxydata]; colours = [coord for xs in colours for coord in xs]
        scales = [galaxy[4] for galaxy in galaxydata]; scales = np.array([coord for xs in scales for coord in xs])
        stars = [x, y, z, colours, scales]
        return stars
    
    def get_blackholes(self):
        ''' Return a list of all of the BlackHole objects in the universe (with the local blackhole being the last element). 
        Returns
        -------
        allblackholes : list
            A list of all of the BlackHole objects in the universe, with the distant blackholes populating the first section of the list,
            and black holes in resolved galaxies populating the second section of the list.
        '''
        blackholes = [galaxy.blackhole for galaxy in self.galaxies]
        distantblackholes = [galaxy.blackhole for galaxy in self.distantgalaxies]
        allblackholes = blackholes + distantblackholes
        return allblackholes
    
    def get_radial_velocities(self):
        ''' Calculates the radial velocities for all of the stars/distant galaxies in the universe, according to:
             - Hubble Recession
             - Galaxy Rotation
             - Cluster Rotation
             - Random rotations of galaxies in 3D space
        Returns
        -------
        obsvel : numpy array
            The radial velocities of the stars in the universe
        distantobsvel : numpy array
            The radial velocities of the distant galaxies
        '''
        stars = self.get_all_starpositions()
        x, y, z, _, _ = stars[0], stars[1], stars[2], stars[3], stars[4]
        _, _, radius = self.cartesian_to_spherical(x, y, z)
        
        locgalaxymovement = self.clusters[-1].directions[:, -1]  # the local galaxy is the last galaxy in the last cluster
        localgalaxydist = self.clusters[-1].galaxies[-1].spherical[2]   # get the distance from the observer (origin) to the center of the local galaxy
        localgalaxy = self.clusters[-1].galaxies[-1]
        # now, we want to find the velocity of a star (in a similar orbit to that of "our sun") about the center of the local galaxy
        closestar = min(localgalaxy.starorbits, key=lambda x:abs(x - localgalaxydist))  # find the star with orbital radius closest to our distance from galax center
        closestarindex = np.where(localgalaxy.starorbits == closestar)  # get the index of that close star
        approxlocalstarvel = localgalaxy.starvels[1, closestarindex[0]]     # find the velocity of that close star, with index 0 in case there are more than one match
        # since the local galaxy is hardcoded to be at (180, 90) coords, we can fix the motion vector, with no x movement, negative y movement, and random z motion
        localstarmovement = approxlocalstarvel * np.array([0, -1, np.random.normal(0, 0.05)])
        
        galaxydirection = []            # initialise lists that will hold the direction vectors for each galaxy and distant galaxy
        distantgalaxydirection = []
        for cluster in self.clusters:       # this for loop finds the direction of galaxy movement within their respective clusters
            for i in range(len(cluster.galaxies)):
                # add the vector of the local galaxy movement with the current galaxy movement
                if cluster.galaxies[i].complexity != "Distant":     # add direction to normal galaxy list if not distant
                    galaxydirection.append(cluster.directions[:, i])    
                else:   # else add the direction to the distant galaxy list
                    distantgalaxydirection.append(cluster.directions[:, i]) 
        galaxydirection = np.array(galaxydirection)     # to help in reading the data later, we want the data to be in an array form rather than a list
        distantgalaxydirection = np.array(distantgalaxydirection)
        DGcartesians = np.array([galaxy.cartesian for galaxy in self.distantgalaxies])    # distant galaxy cartesians coords
        DGx, DGy, DGz = DGcartesians[:, 0], DGcartesians[:, 1], DGcartesians[:, 2]     # distant galaxy x, distant galaxy y, etc
        DGradius = [galaxy.spherical[2] for galaxy in self.distantgalaxies]     # get the distance to each distant galaxy
        
        starvectors = []     # initialise list that will hold star direction vectors for stars within each galaxy
        distantgalaxyvectors = []
        k = 0; m = 0    # these are required to keep track of which galaxy we're dealing with, "close" and "distant" galaxy tickers respectively
        # this for loop calculates the velocity *vector* (magnitude and direction) of each star in the universe, as well as distant galaxies
        for h, cluster in enumerate(self.clusters):     
            galaxyvels = cluster.galaxvels[1, :]    # get the magnitude of the velocity of this galaxy in its cluster
            for i, galaxy in enumerate(cluster.galaxies):   # for the galaxies in this cluster...
                if galaxy.complexity != "Distant":  # close galaxy, so we're dealing with stars
                    stardirection = galaxy.directions   # get the directions of all of the stars in the galaxy
                    starvels = galaxy.starvels[1, :]    # get the magnitude of the velocity of this galaxy's stars
                    galaxyvel = galaxyvels[i]           # recall the velocity of this galaxy
                    for j in range(len(stardirection[0, :])):   # for each star in this galaxy...
                        if h == len(self.clusters) - 1 and i == len(cluster.galaxies) - 1:      # must be the local galaxy
                            vector = (stardirection[:, j] * starvels[j])   # get the vector by multiplying direction by magnitude
                        else:
                            vector = (stardirection[:, j] * starvels[j]) + (galaxydirection[k] * galaxyvel)     # as above, but including the vector inherent with the moving galaxy
                        starvectors.append(vector)  # add this stars vector to the list of all star vectors
                    k += 1      # increase the close galaxy ticker by 1
                else:   # distant galaxy, so we're dealing with galaxy as a whole
                    vector = distantgalaxydirection[m] * galaxyvels[i]  # multiply direction by magnitude
                    distantgalaxyvectors.append(vector)     # add vector the list of distant galaxy vectors
                    m += 1  # increment the distant galaxy ticker
                    
        k = 0; m = 0    # restart the galaxy tickers from above
        obsvel = np.zeros(len(starvectors))
        distantobsvel = np.zeros(len(distantgalaxyvectors))
        for h, cluster in enumerate(self.clusters):
            # this for loop gets the observed radial velocities of all stars and distant galaxies when accounting for their
            # velocity vectors AND the velocity vector inherent of the observer moving about the local galaxy center
            if h != len(self.clusters) - 1:     # if the current cluster is NOT the local cluster...
                clustervel = self.clustervels[h]    
                addclustervel = True    # ...then we want to add this clusters radial velocity onto the vector of any star/distant galaxy
            else:   # if it is the current cluster, the observer is moving with the cluster so we dont add any cluster vector
                addclustervel = False
            for i, galaxy in enumerate(cluster.galaxies):
                if galaxy.complexity != "Distant":  # close galaxy, so we're working with individual stars
                    if addclustervel == False and i == len(cluster.galaxies) - 1:   # this must be our local galaxy! we dont want to add galaxy movement onto this since we're moving *with* the galaxy
                        addgalaxyvel = False
                    else:   # else its not our galaxy, so it has motion relative to our galaxy
                        addgalaxyvel = True
                    for j in range(len(galaxy.stars)):
                        if addgalaxyvel:
                            vector = starvectors[k] + locgalaxymovement + localstarmovement    # velocity vector "v"
                        else:
                            vector = starvectors[k] + localstarmovement
                        coord = np.array([x[k], y[k], z[k]])    # distance vector "d" away from the origin
                        obsvel[k] = np.dot(vector, coord) / radius[k]      # dot product: (v dot d) / ||d||
                        # the dot product above gets the radial component of the velocity (thank you Ciaran!! - linear algebra is hard)
                        if addclustervel:
                            obsvel[k] += clustervel     # if its not our cluster, add the clusters' velocity due to hubble recession
                        k += 1      # increment galaxy ticker
                else:   # distant galaxy, so we're working with galaxies as a whole
                    vector = distantgalaxyvectors[m] + locgalaxymovement + localstarmovement    # very similar process to above
                    coord = np.array([DGx[m], DGy[m], DGz[m]])
                    distantobsvel[m] = np.dot(vector, coord) / DGradius[m]
                    if addclustervel:
                        distantobsvel[m] += clustervel
                    m += 1
        return obsvel, distantobsvel
    
    def explode_supernovae(self, frequency):
        ''' Generate Type 1a supernovae in random galaxies in the universe, with their apparent peak flux.
        Parameters
        ----------
        frequency : int
            The number of supernovae to generate
        Returns
        -------
        skypositions : list
            A list of [equat, polar], where equat and polar are numpy arrays of the coordinates of each supernova in the sky
        peakfluxes : numpy array
            The peak flux (in W/m^2) of the supernovae, accounting for distance to the observer.
        '''
        allgalaxies = self.distantgalaxies + self.galaxies
        indexes = np.random.uniform(0, len(allgalaxies) - 1, frequency - 2)     # choose positions of the supernovae in terms of random galaxy indexes
        closeindexes = len(allgalaxies) - np.random.uniform(1, 14, 2)     # gets two indexes within the last 14 of the galaxy list
        indexes = np.append(indexes, closeindexes); np.random.shuffle(indexes)  # shuffle the indexes so its not as obvious that the last two supernovae are in close galaxies
        galaxies = [allgalaxies[int(i)] for i in indexes]
        positions = np.array([galaxy.spherical for galaxy in galaxies])
        # intrinsic = 1.5 * 10**44 / (4 * np.pi * (7 * 10**6)**2)     # rough energy release of R=7000km white dwarf Type Ia supernova (W/m^2)
        # peakmag = -18.4; sunmag = 4.74; sunlumin = 3.828 * 10**26   # peak magnitude of a type 1a supernova (M_V), bol abs mag of the sun, bol lumin of the sun
        # peaklumin = sunlumin * 10**((peakmag - sunmag) / (-2.5))    # this is the mag/lumin formula rearranged to give L
        # intrinsicflux = peaklumin / (4 * np.pi * (5 * 10**6)**2)    # rough energy release of R=7000km white dwarf Type Ia supernova (W/m^2)
        # intrinsicflux *= (10 * 3.086 * 10**16)**2                   # account for the peakmag being at 10pc
        # distances = positions[:, 2] * 3.086 * 10**16    # convert from parsec to meters
        # peakfluxes = (intrinsicflux / distances**2) * np.random.normal(1, 0.01, frequency)
        
        peaklumin = 2 * 10**36  # 20 billion times solar luminosity, source: https://www.sciencedirect.com/topics/physics-and-astronomy/type-ia-supernovae#:~:text=A%20typical%20supernova%20reaches%20its,times%20that%20of%20the%20Sun.
        distances = positions[:, 2] * 3.086 * 10**16    # convert from parsec to meters
        peakfluxes = (peaklumin / (4 * np.pi * distances**2)) * np.random.normal(1, 0.01, frequency)  # F = L / (4pi*r^2)  - with some random scatter
        skypositions = [positions[:, 0] + np.random.normal(0, 0.01, frequency),   # equatorial angle, with a bit of scatter
                        positions[:, 1] + np.random.normal(0, 0.01, frequency)]   # as above, but with polar angle
        return skypositions, peakfluxes
    
    def plot_hubblediagram(self, trendline=True, save=False):
        ''' Plot the hubble diagram with distance in units of kiloparsec
        Parameters
        ----------
        trendline : bool
            If true, include a trendline indicating the true hubble parameter
        save : bool
            Used in the UniverseSim.save_data() function further downstream. If true, returns the figure to save later. 
        Returns
        -------
        fig : matplotlib figure object
            If save==True, returns the figure to be saved later on. 
        '''
        fig, ax = plt.subplots()
        ax.scatter(self.clusterdists / 1000, self.clustervels, s=1)     # plots the cluster dists in kpc
        ax.set_xlabel("Distance (kpc)"); ax.set_ylabel("Velocity (km/s)")
        
        if trendline:   # plot a trendline 
            x = np.array([0, self.radius])
            y = (self.hubble / 10**6) * x   # get the radial velocity in terms of km/s/pc * pc => km/s
            ax.plot(x / 1000, y, 'r-', alpha=0.5)
        
        ax.set_xlim(xmin=0); ax.set_ylim(ymin=0)
        if save:
            plt.close()
            return fig
        
        
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
            