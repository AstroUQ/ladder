# Simulated Universe

![universe](/MiscGithubImages/universe.jpg)
[![License](https://img.shields.io/badge/License-MIT-brightgreen)](https://spdx.org/licenses/MIT.html)
## About
As part of the University of Queensland course *PHYS3080 - Extragalactic Astrophysics and Cosmology*, we students were given data from a simulated universe (programmed by the course staff) to create an astronomical distance ladder. Both the data analysis and the simulation itself fascinated me, so I'd decided to try my hand at emulating the simulation myself. While some aspects in this simulation aren't as polished as the data we were given as students, some new features have been added, and others extended. I'm really happy with the results so far, and I aim to work on it and tweak the code for a while to come. 

As python is the language I'm most capable in, this project is written entirely in python. Several packages were used extensively, including but not limited to: `numpy`, `matplotlib`, `pandas`, `multiprocessing`, and `scipy`.

## Extent of Simulation
Several real aspects of astronomy are modelled (to varying degrees of accuracy) in this simulation. Below is a non-exhaustive list of the physical properties that could, in theory, be derived via quantitative and qualitative means from the output data `.txt` files:
 - Galaxies:
   - Classification of galaxy types. cD, E0-7, S0 and Sa-c, SBa-c galaxy types are simulated to a reasonable degree of accuracy. 
   - Different star populations are found throughout galaxies. In spirals, the bulge/bar has lower mass, redder stars. The leading edges of spirals have high mass blue stars, with lower mass redder stars trailing the leading edges. Elliptical galaxies are characterised by many more, lower mass red stars.
   - Rotation curves are simulated accurately according to newtonian physics, with black holes and dark matter (via the Navarro-Frenk-White profile) influencing rotation velocities. 


## To-Do:
- [ ] Fill out many docstrings and comment all code to a degree readable by other users.
- [ ] Set it so that some galaxies don't have a black hole and/or dark matter, with probability depending on galaxy type. 
- [ ] Output some luminosity data about black holes. Maybe under x-ray data?
- [ ] Make brightness of black holes depend on angle of the galaxy to the viewer? As well as mass and random activity (possibly depending on galaxy type)

## Credits/Acknowledgements
 - The `blackbodycolours.txt` file was supplied free of charge by Mitchell Charity ([email](mailto:mcharity@lcs.mit.edu)) from [What colour is a blackbody?](http://www.vendian.org/mncharity/dir3/blackbody/) (Version 2001-Jun-22)
