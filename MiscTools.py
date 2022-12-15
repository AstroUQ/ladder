# -*- coding: utf-8 -*-
"""
Created on Fri Nov 25 19:16:37 2022

@author: ryanw
"""
import numpy as np
import matplotlib.pyplot as plt

def cartesian_to_spherical(x, y, z):
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

def spherical_to_cartesian(equat, polar, distance):
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

def cartesian_rotation(angle, axis):
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
    
def cubemap(x, y, z):
    ''' Transforms cartesian coordinates into (u, v) coordinates on the 6 faces of a cube, with an index corresponding to 
        which cube face the point is projected on to. 
    Parameters
    ----------
    x, y, z : np.array
        Cartesian coordinates of the point(s)
    Returns
    -------
    uc, vc : np.array
        horiz, vertical coords (respectively) in the corresponding cube face
    index : np.array
        corresponding cube face index of each projected point in (x, y, z):
        {0: front, 1: back, 2: top, 3: bottom, 4: left, 5: right}
    '''
    # initialise arrays
    index = np.zeros(x.size, dtype=int); uc = np.zeros(x.size); vc = np.zeros(x.size)
    # rotate the points so that the local galactic center is centered in the 'front' image
    # that is, equat=180 => front, uc=0
    points = np.array([x, y, z])
    points = np.dot(cartesian_rotation(np.pi, 'y'), points)
    points = np.dot(cartesian_rotation(np.pi / 2, 'x'), points)
    x, y, z = points
    
    # now, let's find which cube face each point is projected to. This algorithm was taken from wikipedia, and adapted
    # for python: https://en.wikipedia.org/wiki/Cube_mapping#Memory_addressing
    for i in range(x.size):
        if x.size == 1: # gotta account for arrays of one value
            X = x; Y = y; Z = z
        else:
            X = x[i]; Y = y[i]; Z = z[i]
        # normalise each vector component so that the output coords are between -x and +x (for example)
        absArray = abs(np.array([X, Y, Z]))
        X /= max(absArray); Y /= max(absArray); Z /= max(absArray)
        # now we can find which cube face the point is projected onto:
        if X > 0 and abs(X) >= abs(Y) and abs(X) >= abs(Z): # point is on: POS X -- front
            uc[i] = -Z
            vc[i] = Y
        elif X < 0 and abs(X) >= abs(Y) and abs(X) >= abs(Z): # NEG X -- back
            uc[i] = Z
            vc[i] = Y
            index[i] = 1
        elif Y > 0 and abs(Y) >= abs(X) and abs(Y) >= abs(Z): # POS Y -- top
            uc[i] = X
            vc[i] = -Z
            index[i] = 2
        elif Y < 0 and abs(Y) >= abs(X) and abs(Y) >= abs(Z): # NEG Y -- bottom
            uc[i] = X
            vc[i] = Z
            index[i] = 3
        elif Z > 0 and abs(Z) >= abs(X) and abs(Z) >= abs(Y): # POS Z -- left
            uc[i] = X
            vc[i] = Y
            index[i] = 4
        else: # Z < 0 and abs(Z) >= abs(X) and abs(Z) >= abs(Y)  # NEG Z -- right
            uc[i] = -X
            vc[i] = Y
            index[i] = 5
            
    uc *= 45; vc *= 45 # transforms coords from +/- 1 to +/- 45 degrees
    return uc, vc, index

def gen_figAxes(method="AllSky"):
    ''' Generates figures in a format applicable for plotting allsky or cubemapped data (as you'd expect from a telescope image).
    Produces a black background (because space is black, duh), of the appropriate axis limits and length ratio.
    Parameters
    ----------
    method : str
        Method for creating the figures. One of {"AllSky", "Cube"}, where the output list will be of length 1 and 6 respectively.
    Returns
    -------
    figAxes : list
        List of one or more lists, formatted as [[fig1, ax1], [fig2, ax2],...] for matplotlib figure and axes objects
    '''
    if method=="AllSky":
        fig, ax = plt.subplots()
        ax.invert_yaxis()
        ax.set_facecolor('k')
        ax.set_aspect('equal')
        figAxes = [[fig, ax]]
    else:
        figAxes = []
        for i in range(6):
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.set_xlim(-45, 45); ax.set_ylim(-45, 45)    # equatorial angle goes from 0->360, polar 0->180
            ax.set_facecolor('k')   # space has a black background, duh
            ax.set_aspect(1)    # makes it so that the figure is twice as wide as it is tall - no stretching!
            # fig.tight_layout()
            ax.set_xlabel("X Position (degrees)")
            ax.set_ylabel("Y Position (degrees)")
            ax.grid(linewidth=0.1)
            figAxes.append([fig, ax])
    return figAxes