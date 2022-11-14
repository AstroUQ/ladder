# -*- coding: utf-8 -*-
"""
Created on Fri Sep 16 17:21:53 2022

@author: ryanw
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

class Nebula(object):
    def __init__(self, species, position, radius=None, cartesian=False, rotation=None, localgalaxy=False):
        self.species = species
        self.radius = radius
        self.local = localgalaxy
        self.nebula_params()
        self.rotation = rotation if rotation is not None else np.random.uniform(0, 2*np.pi, 3)
        self.cmap = self.initColourMap(self.palette)
        self.cartesian = position if cartesian == True else self.spherical_to_cartesian(position[0], position[1], position[2])
        self.spherical = self.cartesian_to_spherical(position[0], position[1], position[2]) if cartesian == True else position
        self.points = self.gen_points()
        
    def nebula_params(self):
        if self.species == "ring":
            self.radii = np.array([0.15, 0.12, 0.08]) * 5 if self.radius == None else self.radius
            self.palette = "Hubble"
        elif self.species[0] == "S":
            self.radii = self.radius
            self.palette = 'Spiral'
    
    def initColourMap(self, palette):
        N = 256
        if palette == "Hubble":
            valsR = np.ones((N, 4))
            valsG = np.ones((N, 4))
            valsB = np.ones((N, 4))
            midval = 180
            # start with hubble blue
            valsB[:, 0] = 0
            valsB[:midval, 1] = 178 / 256; valsB[midval:, 1] = np.linspace(178 / 256, 43/256, N - midval)
            valsB[:midval, 2] = 255 / 256; valsB[midval:, 2] = np.linspace(255 / 256, 29/256, N - midval)
            valsB[20:, 3] = np.linspace(0, 0.6, N - 20); valsB[:20, 3] = 0
            HubbleBlue = ListedColormap(valsB)
            # LinearSegmentedColormap('HubbleBlue', valsB)
    
            valsG[:midval, 0] = 200 / 256; valsG[midval:, 0] = np.linspace(200 / 256, 66/256, N - midval)
            valsG[:midval, 1] = 255 / 256; valsG[midval:, 1] = np.linspace(255 / 256, 43/256, N - midval)
            valsG[:, 2] = 0
            valsG[20:, 3] = np.linspace(0, 0.6, N - 20); valsG[:20, 3] = 0
            HubbleGreen = ListedColormap(valsG)
            # LinearSegmentedColormap('HubbleGreen', valsG)
    
            valsR[:midval, 0] = 255 / 256; valsR[midval:, 0] = np.linspace(255 / 256, 66/256, N - midval)
            valsR[:, 1] = 0
            valsR[:, 2] = 0
            valsR[20:, 3] = np.linspace(0, 0.6, N - 20); valsR[:20, 3] = 0
            HubbleRed = ListedColormap(valsR)
            # LinearSegmentedColormap('HubbleRed', valsR)
            
            colourmap = [HubbleRed, HubbleGreen, HubbleBlue]
        elif palette == 'Spiral':
            # vals = np.zeros((N, 4))
            # vals[:, 0] = np.linspace(99, 238, N) / 256
            # vals[:, 1] = np.linspace(99, 228, N) / 256
            # vals[:, 2] = np.linspace(125, 214, N) / 256
            # vals[20:60, 3] = np.linspace(0, 0.2, 60-20); vals[60:, 3] = 0.2
            # SpiralGalax = ListedColormap(vals)
            # # LinearSegmentedColormap('SpiralGalax', vals)
            # colourmap = [SpiralGalax]
            valsDisk = np.zeros((N, 4))
            valsBulge = np.zeros((N, 4))
            valsArms = np.zeros((N, 4))
            
            midval = 100
            alphamid = 20
            valsDisk[:, 0] = 102 / 256; valsDisk[midval:, 0] = np.linspace(166 / 256, 43/256, N - midval)
            valsDisk[:midval, 1] = 96 / 256; valsDisk[midval:, 1] = np.linspace(134 / 256, 43/256, N - midval)
            valsDisk[:midval, 2] = 129 / 256; valsDisk[midval:, 2] = np.linspace(155 / 256, 29/256, N - midval)
            valsDisk[alphamid:, 3] = np.linspace(0, 0.1, N - alphamid); valsDisk[:alphamid, 3] = 0
            SpiralDisk = ListedColormap(valsDisk)
            
            valsBulge[:, 0] = 229 / 256; valsBulge[midval:, 0] = np.linspace(229 / 256, 253 / 256, N - midval)
            valsBulge[:midval, 1] = 199 / 256; valsBulge[midval:, 1] = np.linspace(199 / 256, 244/256, N - midval)
            valsBulge[:midval, 2] = 189 / 256; valsBulge[midval:, 2] = np.linspace(189 / 256, 239/256, N - midval)
            valsBulge[alphamid:, 3] = np.linspace(0, 0.3, N - alphamid); valsBulge[:alphamid, 3] = 0
            SpiralBulge = ListedColormap(valsBulge)
            
            valsArms[:, 0] = 183 / 256
            valsArms[:, 1] = 185 / 256
            valsArms[:, 2] = 240 / 256
            valsArms[alphamid:, 3] = np.linspace(0, 0.8, N - alphamid); valsArms[:alphamid, 3] = 0
            SpiralArms = ListedColormap(valsArms)
            
            if self.species[1] == "B":
                valsBar = np.zeros((N, 4))
                valsBar[:, 0] = 229 / 256
                valsBar[:, 1] = 199 / 256
                valsBar[:, 2] = 189 / 256
                valsBar[alphamid:, 3] = np.linspace(0, 0.6, N - alphamid); valsBar[:alphamid, 3] = 0
                SpiralBar = ListedColormap(valsBar)
                colourmap = [SpiralDisk, SpiralBulge, SpiralArms, SpiralBar]
            else:
                colourmap = [SpiralDisk, SpiralBulge, SpiralArms]
            
        return colourmap
    
    def gen_points(self):
        if self.species == 'ring':
            points = self.gen_ring_nebula()
        elif self.species[0] == "S":   # dealing with spiral galaxy
            points = self.gen_spiral_nebulosity()
            
        return points
    
    def gen_ring_nebula(self):
        pops = [150000, 120000, 100000]
        lowers = [0, 0.1, 0.2]
        coords = []
        for i, colour in enumerate(self.cmap):
            pop = pops[i]
            radius = self.radii[i] * (np.random.uniform(lowers[i], 1, pop))**(1/3)
            # the following distributes points more or less evenly about a sphere centered at the origin
            theta = np.random.uniform(0, 2*np.pi, pop)
            vert = 0.6 if i < 2 else 1
            phi = np.arccos(np.random.uniform(-vert, vert, pop))
            centerx = radius * (np.cos(theta) * np.sin(phi) * np.random.normal(1, 0.1, pop))
            centery = radius * (np.sin(theta) * np.sin(phi) * np.random.normal(1, 0.1, pop))
            zmult = 1 if i < 2 else 1.5
            centerz = zmult * radius * (np.cos(phi) * np.random.normal(1, 0.3, pop))
            if i == 0:
                outPop = int(0.75 * pop); outRad = 1.5 * self.radii[i]
                outPhi = np.arccos(np.random.uniform(-1, 1, outPop))
                outTheta = np.random.uniform(0, 2*np.pi, outPop)
                centerx = np.append(centerx, outRad * (np.cos(outTheta) * np.sin(outPhi) * np.random.normal(1, 0.1, outPop)))
                centery = np.append(centery, outRad * (np.sin(outTheta) * np.sin(outPhi) * np.random.normal(1, 0.1, outPop)))
                centerz = np.append(centerz, outRad * (np.cos(outPhi) * np.random.normal(1, 0.3, outPop)))
            
            points = np.array([centerx, centery, centerz])
            points = np.dot(self.nebula_rotation(self.rotation[0], 'x'), points)
            points = np.dot(self.nebula_rotation(self.rotation[1], 'y'), points)
            points = np.dot(self.nebula_rotation(self.rotation[2], 'z'), points)
            
            points[0] += self.cartesian[0]
            points[1] += self.cartesian[1]
            points[2] += self.cartesian[2]
            coords.append(points)
        return coords
    
    def gen_spiral_nebulosity(self):
        coords = []
        for i, colour in enumerate(self.cmap):
            pop = 100000
            pop = int(3e5)
            
            theta = np.random.uniform(0, 2*np.pi, pop)
            # phi = np.random.uniform(0, 2*np.pi, pop)
            if i == 0:      # we're dealing with the disk component
                # diskdists = np.random.exponential(self.radius/2, size=pop)
                diskdists = self.radius * np.random.uniform(0, 1, size=pop)**(1/2)
                x = np.cos(theta) * diskdists * np.random.normal(1, 0.2, pop)
                y = np.sin(theta) * diskdists * np.random.normal(1, 0.2, pop)
                z = np.zeros(pop) + 0.02 * self.radius * np.random.randn(pop)
            elif i == 1:       # we're dealing with the bulge
                bulgepop = pop
                
                # theta = np.random.uniform(0, 2*np.pi, bulgepop)
                costheta = np.random.uniform(-1, 1, bulgepop)
                theta = np.arccos(costheta)
                
                phi = np.random.uniform(0, 2*np.pi, bulgepop)
                bulgeradius = 0.4 * self.radius
                bulgeR = bulgeradius * np.random.uniform(0, 1, bulgepop)**(1/2)    #bulgedists was meant to be RVs between 0 and 1, but the mult makes up for it
                x = bulgeR * (np.cos(theta) * np.sin(phi) + np.random.normal(0, 0.1, bulgepop))
                y = bulgeR * (np.sin(theta) * np.sin(phi) + np.random.normal(0, 0.1, bulgepop))
                distanceflat = (1 / self.radius) * np.sqrt(np.square(x) + np.square(y))     #this makes the z lower for stars further from the center
                z = 0.5 * bulgeR * np.cos(phi) + np.random.normal(0, 0.1, bulgepop) #* 0.8**distanceflat
                # z = np.sqrt(0.6**2 * (bulgeradius**2 - (x**2 + y**2))) #* np.random.normal(0, 1, bulgepop)
                
                
                # x = np.cos(theta) * np.sin(phi) * np.random.uniform(0, 1, bulgepop)**(1/3)
                # y = np.sin(theta) * np.sin(phi) * np.random.uniform(0, 1, bulgepop)**(1/3)
                # z =  np.sqrt(0.4**2 * (1 - (x**2 + y**2))) * np.random.choice([-1, 1], bulgepop)#* np.random.normal(0, 1, bulgepop)
                # x *= bulgeradius #* np.random.normal(0, 0.5, bulgepop)
                # y *= bulgeradius #* np.random.normal(0, 0.5, bulgepop)
                # z *= bulgeradius * np.random.uniform(0, 1, bulgepop)**(1/3) #* np.random.normal(0, 0.5, bulgepop)
                
                
                # bulgeradius = 0.4 * self.radius
                # bulgeR = bulgeradius * np.random.uniform(0, 1, bulgepop)**(3/5)
                # x = np.cos(theta) * bulgeR * np.random.normal(1, 0.2, bulgepop)
                # z = np.sin(theta) * bulgeR * np.random.normal(1, 0.2, bulgepop)
                # y = np.zeros(bulgepop) + 0.02 * self.radius * np.random.randn(bulgepop)
                
            elif i in [2, 3]:       # we're dealing with spiral arms
                pop = 10000
                speciesindex = {"S0":0, "Sa":1, "Sb":2, "Sc":3, "SBa":4, "SBb": 5, "SBc":6}
                wrap = [[None, None], [0.9, 4 * np.pi], [0.7, 2 * np.pi], [0.2, 0.8 * np.pi], 
                        [np.pi / 2.1, 3 * np.pi], [np.pi / 2.1, 2 * np.pi], [np.pi / 2.1, 1.15 * np.pi]]
                
                if i == 2:
                    #now to actually grab the parameters for the galaxy type in question:
                    SpiralRadiiDiv = [None, 15, 7, 2.1, 3.7, 3, 2.3] 
                    mult, spiralwrap = [param[speciesindex[self.species]] for param in [SpiralRadiiDiv, wrap]]
                    upper, lower = spiralwrap
                    if speciesindex[self.species] >= 5:
                        spiralangle = np.geomspace(lower, upper, pop)
                    else:
                        spiralangle = np.linspace(lower, upper, pop)
                    reflect = np.random.choice([-1, 1], pop)
                    # power = 1/2 if self.species[:2] == "SB" else 1
                    spiralpow = np.sqrt(spiralangle) if self.species[:2] == "SB" else spiralangle
                    [lag, scatter, scatter2, zscatter] = [0, 0.1, 0.1, 0.03]
                    x = (self.radius / mult) * (spiralpow * np.cos(spiralangle + lag)  * np.random.normal(1, scatter, pop) * reflect + np.random.normal(0, scatter2, pop))
                    y = (self.radius / mult) * (spiralpow * np.sin(spiralangle + lag) * np.random.normal(1, scatter, pop) * - reflect + np.random.normal(0, scatter2, pop))
                    z = np.random.normal(0, zscatter * self.radius, pop)
                elif i == 3:    # we're dealing with spiral bar
                    barradii = [0, 0, 0, 0, 0.3, 0.4, 0.5]
                    barradius = barradii[speciesindex[self.species]] * self.radius
                    x = np.random.normal(0, 0.3 * barradius, pop)
                    # y = barradius * (np.geomspace(0.3, 1.1, pop) * np.random.choice([-1, 1], pop) + np.random.normal(0, 0.3, pop))
                    y = barradius * (np.linspace(0, 1.1, pop) * np.random.choice([-1, 1], pop) + np.random.normal(0, 0.3, pop))
                    z = np.random.normal(0, 0.1 * barradius, pop)
            
            points = np.array([x, y, z])
            
            # if i != 1:
            #     points = np.dot(self.nebula_rotation(self.rotation[0], 'x'), points)
            #     points = np.dot(self.nebula_rotation(self.rotation[1], 'y'), points)
            #     points = np.dot(self.nebula_rotation(self.rotation[2], 'z'), points)
            points = np.dot(self.nebula_rotation(self.rotation[0], 'x'), points)
            points = np.dot(self.nebula_rotation(self.rotation[1], 'y'), points)
            points = np.dot(self.nebula_rotation(self.rotation[2], 'z'), points)
            
            
            # x = np.append(diskx, bulgex); y = np.append(disky, bulgey); z = np.append(diskz, bulgez)

            # points = [np.append(points[0], bulgex), np.append(points[1], bulgey),np.append(points[2], bulgez)]
            
            points[0] += self.cartesian[0]
            points[1] += self.cartesian[1]
            points[2] += self.cartesian[2]
            coords.append(points)
        return coords
        
        
    def plot_nebula(self, ax=None, style='colormesh'):
        if self.species == 'spiral':
            bins = 200
            grid = 120
        else:
            bins = 150
            grid = 50
        if ax == None:
            fig, ax = plt.subplots()
            ax.invert_yaxis()
            ax.set_facecolor('k')
            ax.set_aspect('equal')
        if style in ['colormesh', 'imshow']:
            for i, colour in enumerate(self.cmap):
                vmax = None
                    
                if self.palette == 'Spiral':
                    smooth = 2.5
                else:
                    smooth = 1.5
  
                x, y, z = self.points[i]
                equat, polar, distance = self.cartesian_to_spherical(x, y, z)
                    
                extent = [[min(equat) - 3, max(equat) + 3], [min(polar) - 3, max(polar) + 3]]   # this is so that the edge of the contours aren't cut off
                density, equatedges, polaredges = np.histogram2d(equat, polar, bins=[2 * bins, bins], range=extent)
                equatbins = equatedges[:-1] + (equatedges[1] - equatedges[0]) / 2   # this fixes the order of the bins, and centers the bins at the midpoint
                polarbins = polaredges[:-1] + (polaredges[1] - polaredges[0]) / 2
                
                density = density.T      # take the transpose of the density matrix
                density = scipy.ndimage.zoom(density, 2)    # this smooths out the data so that it's less boxy and more curvey
                equatbins = scipy.ndimage.zoom(equatbins, 2)
                polarbins = scipy.ndimage.zoom(polarbins, 2)
                # if self.palette == 'Spiral:'
                density = scipy.ndimage.gaussian_filter(density, sigma=smooth)  # this smooths the area density even moreso (not necessary, but keeping for posterity)
                
                if style == 'colormesh':
                    # import matplotlib.colors as colors
                    # ax.pcolormesh(equatbins, polarbins, density, cmap=colour, shading='auto', rasterized=True, 
                    #               norm=colors.PowerNorm(gamma=0.8))
                    ax.pcolormesh(equatbins, polarbins, density, cmap=colour, vmax=vmax, shading='auto', rasterized=True, antialiased=True)
                elif style == 'imshow':
                    extent = [min(equat), max(equat), min(polar), max(polar)]
                    ax.imshow(density, extent=extent, cmap=colour, interpolation='none')
        elif style == 'hexbin':
            for i, colour in enumerate(self.cmap):
                x, y, z = self.points[i]
                equat, polar, distance = self.cartesian_to_spherical(x, y, z)
                if self.species == 'spiral':
                    ax.hexbin(equat, polar, gridsize=(2 * grid, grid), bins=bins, vmax = 6, linewidths=0.01, cmap=colour, aa=True)
                else:
                    ax.hexbin(equat, polar, gridsize=(2 * grid, grid), bins=bins, linewidths=0.01, cmap=colour)

    def nebula_rotation(self, angle, axis):
        '''Rotate a point in cartesian coordinates about the origin by some angle along the specified axis. 
        The rotation matrices were taken from https://stackoverflow.com/questions/34050929/3d-point-rotation-algorithm
        Parameters
        ----------
        angle : float
            An angle in radians.
        axis : str
            The axis to perform the rotation on. Must be in ['x', 'y', 'z']
        Returns
        -------
        numpy array
            The transformation matrix for the rotation of angle 'angle'. This output must be used as the first argument within "np.dot(a, b)"
            where 'b' is an 3 dimensional array of coordinates.
        '''
        if axis == 'x':
            return np.array([[1, 0, 0], [0, np.cos(angle), -np.sin(angle)], [0, np.sin(angle), np.cos(angle)]])
        elif axis == 'y':
            return np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]])
        else:
            return np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])

    def cartesian_to_spherical(self, x, y, z):
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
        equat, polar, distance : numpy arrays
            equatorial and polar angles, as well as radial distance from the origin
        Returns
        -------
        (x, y, z) : numpy arrays
            Cartesian coordinates relative to the origin. 
        '''
        equat, polar = np.radians(equat), np.radians(polar)
        x = distance * np.cos(equat) * np.sin(polar)
        y = distance * np.sin(equat) * np.sin(polar)
        z = distance * np.cos(polar)
        return (x, y, z)




# img = plt.imread("Datasets/Sim Data (Clusters; 1000, Seed; 588)/Universe Image.png")


# ax.imshow(img, extent=[0, 360, 0, 180])

def main():
    # ringNeb = Nebula('ring', [45, 90, 10])
    # ringNeb.plot_nebula(style='hexbin')
    from Galaxy import Galaxy
    position = [45, 90, 1000]
    species = 'SBa'
    galax = Galaxy(species, position)
    fig, ax = plt.subplots()
    
    spiralNeb = Nebula(species, position, galax.radius, rotation=galax.rotation)
    spiralNeb.plot_nebula(style='colormesh', ax=ax)
    galax.plot_2d(fig, ax)
    ax.set_xlim(40, 50)
    ax.set_ylim(95, 85)
    
    # species = 'SBa'
    # position = [180, 90, 40]
    # galax = Galaxy(species, position, rotate=False)
    # fig, ax = plt.subplots()
    
    # spiralNeb = Nebula(species, position, galax.radius, rotation=galax.rotation, localgalaxy=True)
    # spiralNeb.plot_nebula(style='colormesh', ax=ax)
    
    # # ringNeb = Nebula('ring', [150, 85, 10])
    # # ringNeb.plot_nebula(ax=ax)
    
    # galax.plot_2d(fig, ax)
    # fig.set_size_inches(18, 9, forward=True)
    # fig.savefig('galax.png', dpi=1500, bbox_inches='tight', pad_inches = 0.01)
    
    
    
    
if __name__ == "__main__":
    main()




