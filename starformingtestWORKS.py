import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import warnings
warnings.filterwarnings("error")

def cartesian_to_spherical(x, y, z):
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

def spherical_to_cartesian(equat, polar, distance):
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

N = 256
valsR = np.ones((N, 4))
valsG = np.ones((N, 4))
valsB = np.ones((N, 4))
valsW = np.ones((N, 4))
midval = 180
# start with hubble blue
valsB[:, 0] = 134 / 256
valsB[:, 1] = 147 / 256
valsB[:, 2] = 173 / 256
valsB[20:, 3] = np.linspace(0, 0.4, N - 20); valsB[:20, 3] = 0
Blue = ListedColormap(valsB)
# LinearSegmentedColormap('HubbleBlue', valsB)

valsG[:, 0] = 219 / 256
valsG[:, 1] = 169 / 256
valsG[:, 2] = 204 / 256
valsG[20:, 3] = np.linspace(0, 0.4, N - 20); valsG[:20, 3] = 0
Pink = ListedColormap(valsG)
# LinearSegmentedColormap('HubbleGreen', valsG)

valsR[:midval, 0] = 255 / 256
valsR[:, 1] = 0
valsR[:, 2] = 0
valsR[20:, 3] = np.linspace(0, 0.1, N - 20); valsR[:20, 3] = 0
Red = ListedColormap(valsR)
# LinearSegmentedColormap('HubbleRed', valsR)

valsW[:, 0] = 255 / 256
valsW[:, 1] = 255 / 256
valsW[:, 2] = 255 / 256
valsW[20:, 3] = np.linspace(0, 0.7, N - 20); valsW[:20, 3] = 0
White = ListedColormap(valsR)

colourmap = [Red, Pink, Blue, White]






position = np.array([90, 90, 200])
position = spherical_to_cartesian(90, 90, 2000)
radius = 1
segments = np.random.randint(1, 6)
positions = np.ndarray((segments, 3))

for i in range(segments): # generate positions of each segment
    costheta = np.random.uniform(-1, 1)
    theta = np.arccos(costheta)
    
    phi = np.random.uniform(0, 2*np.pi)
    R = radius * np.random.uniform(0, 1)**(1)
    x = R * np.cos(theta) * np.sin(phi)
    y = R * np.sin(theta) * np.sin(phi)
    z = R * np.cos(phi)
    
    x += position[0]; y += position[1]; z += position[2]
    positions[i] = np.array([x, y, z])
    
randomsteps = 2000
population = 100
totalCoords = np.ndarray((len(colourmap), population * randomsteps, 3))

divisor = [1, 5, 10, 20]

fig, ax = plt.subplots()
ax.invert_yaxis()
ax.set_facecolor('k')

for c, colour in enumerate(colourmap):
    
    pops = np.zeros(segments, dtype=int)
    for i in range(segments):
        if i != segments - 1:
            segpop = int(population * np.random.uniform(0, 1 / (i + 1)))
            pops[i] = segpop if segpop < (population - sum(pops)) else (population - sum(pops)) / 2
        else:
            pops[i] = population - sum(pops)
    # if c == 0:
    #     for i in range(segments):
    #         if i != segments - 1:
    #             segpop = int(population * np.random.uniform(0, 1 / (i + 1)))
    #             pops[i] = segpop if segpop < (population - sum(pops)) else (population - sum(pops)) / 2
    #         else:
    #             pops[i] = population - sum(pops)
            
    
    stepsize = (radius / randomsteps) / divisor[c]
    coords = np.ndarray((randomsteps * population, 3))
    
    # steps = np.random.uniform(-stepsize, stepsize, size=(population * randomsteps, 3))
    steps = np.random.normal(0, stepsize, size=(population * randomsteps, 3))
    X, Y, Z = np.zeros(randomsteps * sum(pops)), np.zeros(randomsteps * sum(pops)), np.zeros(randomsteps * sum(pops))
    iteration = 0
    for i in range(segments): # now to do a random walk for each of the points
        # if c == 0:
        #     costheta = np.random.uniform(-1, 1, pops[i])
        #     theta = np.arccos(costheta)
            
        #     phi = np.random.uniform(0, 2*np.pi, pops[i])
        #     R = radius / 3 * np.random.uniform(0, 1, pops[i])**(1/3)
        #     startX = R * np.cos(theta) * np.sin(phi)
        #     startY = R * np.sin(theta) * np.sin(phi)
        #     startZ = R * np.cos(phi)
        #     startX += positions[i][0]; startY += positions[i][1]; startZ += positions[i][2]
        costheta = np.random.uniform(-1, 1, pops[i])
        theta = np.arccos(costheta)
        
        phi = np.random.uniform(0, 2*np.pi, pops[i])
        R = radius / 3 * np.random.uniform(0, 1, pops[i])**(1/3)
        startX = R * np.cos(theta) * np.sin(phi)
        startY = R * np.sin(theta) * np.sin(phi)
        startZ = R * np.cos(phi)
        startX += positions[i][0]; startY += positions[i][1]; startZ += positions[i][2]
        
        for p in range(pops[i]):
            x, y, z = startX[p], startY[p], startZ[p]
            
            for t in range(randomsteps):
                x += steps[iteration, 0]
                y += steps[iteration, 1]
                z += steps[iteration, 2]
                # coords[iteration] = [x, y, z]
                X[iteration] = x
                Y[iteration] = y
                Z[iteration] = z
                totalCoords[c, iteration, 0] = x
                totalCoords[c, iteration, 1] = y
                totalCoords[c, iteration, 2] = z
                iteration += 1
                
    vmax = None
    smooth = 4
    bins = 30
    zoom = 2
    
    # x = coords[:, 0]; y = coords[:, 1]; z = coords[:, 2]
    # equat, polar, distance = cartesian_to_spherical(x, y, z)
    equat, polar, distance = cartesian_to_spherical(X, Y, Z)
    dEquat = (max(equat) - min(equat)); dPolar = (max(polar) - min(polar))
    extent = [[min(equat) - 0.3 * dEquat, max(equat) + 0.3 * dEquat], 
              [min(polar) - 0.3 * dPolar, max(polar) + 0.3 * dPolar]]   # this is so that the edge of the contours aren't cut off
    density, equatedges, polaredges = np.histogram2d(equat, polar, bins=[2 * bins, bins], range=extent)
    # density, equatedges, polaredges = np.histogram2d(equat, polar, bins=[2 * bins, bins])
    equatbins = equatedges[:-1] + (equatedges[1] - equatedges[0]) / 2   # this fixes the order of the bins, and centers the bins at the midpoint
    polarbins = polaredges[:-1] + (polaredges[1] - polaredges[0]) / 2
    
    density = density.T      # take the transpose of the density matrix
    density = scipy.ndimage.zoom(density, zoom)    # this smooths out the data so that it's less boxy and more curvey
    equatbins = scipy.ndimage.zoom(equatbins, zoom)
    polarbins = scipy.ndimage.zoom(polarbins, zoom)
    # if self.palette == 'Spiral:'
    density = scipy.ndimage.gaussian_filter(density, sigma=smooth)  # this smooths the area density even moreso (not necessary, but keeping for posterity)
    
    ax.pcolormesh(equatbins, polarbins, density, cmap=colour, vmax=vmax, shading='auto')



