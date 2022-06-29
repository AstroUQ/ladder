# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 13:08:36 2022

@author: ryanw
"""

from GalaxyCluster import GalaxyCluster
import numpy as np
import math

def progress_bar(progress, total):
    ''' Taken from https://youtu.be/x1eaT88vJUA
    '''
    percent = 100 * (progress / float(total))
    bar = '█' * int(percent / 2) + '-' * (100//2 - int(percent / 2))
    print(f"\r|{bar}| {percent:.2f}%", end='\r')

## this shows how to use the progressbar!
# numbers = [x * 5 for x in range(2000, 3000)]
# results = []

# progress_bar(0, len(numbers))
# for i, x in enumerate(numbers):
#     results.append(math.factorial(x))
#     progress_bar(i + 1, len(numbers))

class Universe(object):
    def __init__(self):
        self.clusterpop = 10
        self.clusters = self.generate_clusters()
    
    def generate_clusters(self):
        progress_bar(0, self.clusterpop)
        clusters = []
        for i in range(self.clusterpop):
            equat = np.random.uniform(0, 360)
            polar = np.random.uniform(0, 180)
            dist = np.random.uniform(1000, 100000)
            pos = (equat, polar, dist)
            pop = int(np.random.exponential(5))
            pop = pop if pop >= 1 else 1
            if dist > 50000:
                clusters.append(GalaxyCluster(pos, pop, complexity="Distant"))
            else:
                clusters.append(GalaxyCluster(pos, pop))
            progress_bar(i+1, self.clusterpop)
        return clusters