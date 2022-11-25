# -*- coding: utf-8 -*-
"""
Created on Fri Nov 25 19:16:37 2022

@author: ryanw
"""
import numpy as np

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