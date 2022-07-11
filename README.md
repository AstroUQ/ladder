# Simulated Universe

![universe](/MiscGithubImages/universe.jpg)
[![License](https://img.shields.io/badge/License-MIT-brightgreen)](https://spdx.org/licenses/MIT.html)
[![GitHub issues](https://img.shields.io/github/issues/ryanwhite1/Simulated-Universe)](https://github.com/ryanwhite1/Simulated-Universe/issues)
[![GitHub stars](https://img.shields.io/github/stars/ryanwhite1/Simulated-Universe)](https://github.com/ryanwhite1/Simulated-Universe/stargazers)
## About
As part of the University of Queensland course *PHYS3080 - Extragalactic Astrophysics and Cosmology*, we students were given data from a simulated universe (programmed by the course staff) to create an astronomical distance ladder. Both the data analysis and the simulation itself fascinated me, so I'd decided to try my hand at emulating the simulation myself. While some aspects in this simulation aren't as polished as the data we were given as students, some new features have been added, and others extended. I'm really happy with the results so far, and I aim to work on it and tweak the code for a while to come. 

As python is the language I'm most capable in, this project is written entirely in python. Several packages were used extensively, including but not limited to: `numpy`, `matplotlib`, `pandas`, `multiprocessing`, and `scipy`.

## Extent of Simulation
Several real aspects of astronomy are modelled (to varying degrees of accuracy) in this simulation. Below is a non-exhaustive list of the physical properties that could, in theory, be derived via quantitative and qualitative means from the output data `.txt` files. In the likely case that I can't follow that theme, the dot points show what and how some things were simulated.
### Stars
<img align="right" height="350" src="/MiscGithubImages/Local Galaxy HR Diagram.png">  

 - Four distinct classes of stars are generated: Main Sequence, Giants, Supergiants, and White Dwarfs. They can be see on the HR diagram on the right in their usual positions.
   - Main sequence stars are generated realistically according to luminosity-mass functions (with the mass first randomly generated via a gamma distribution), and then the radius generated according to mass-radius relations. The temperature is then solved via the Stefan-Boltzmann equation. I am quite happy with the behaviour of the MS stars!
   - White dwarf mass distributions were generated according to a published paper on the topic. Radius was then generated by the $R \sim M^{1/3}$ relation and temperature derived from wikipedia data. The luminosity then was taken via the Stefan-Boltzmann law. 
   - The giant and supergiant parameter generation was far less realistic and was artificially chosen to give nice looking data on a HR diagram.
  
<img align="right" height="200" src="/MiscGithubImages/StarLightCurve.png">  

- Some populations of stars, depending on their position on the HR diagram, will have variable light curves. The simulation will determine whether to have two or three distinct variable populations, with a "short" (18-30 hour) period and a "long" (45-60 hour period) being certain. A "longest" (75-100 hour) period class of variable has a 1/3 chance of being generated too. Each of these classes of variables are shown as triangles in the HR diagram above, with the top population being the short period, the rightmost population being the longest period, and the bottom left being the long period. 
  - The lightcurve of variable stars is randomly chosen out of a Sine, Triangle, or Sawtooth wave, with some random scatter added onto the 'normalised flux'.
  - The period-luminosity relation (linear trend on a logY-x scale) has a randomly assigned sign with a random gradient within some small range of the desired parameters. This is generated at the universe level in `Universe.py`.

### Galaxies
<p align="middle">
<img src="/MiscGithubImages/cD.jpg" height="100"><img src="/MiscGithubImages/ellipticals.jpg" height="100"><img src="/MiscGithubImages/S0.jpg" height="100"><img src="/MiscGithubImages/Sa.jpg" height="100"><img src="/MiscGithubImages/SBa.jpg" height="100"><img src="/MiscGithubImages/Sb.jpg" height="100"><img src="/MiscGithubImages/SBb.jpg" height="100"><img src="/MiscGithubImages/Sc.jpg" height="100"><img src="/MiscGithubImages/SBc.jpg" height="100"> </p>  

 - Classification of galaxy types. cD, E0-7, S0 and Sa-c, SBa-c galaxy types are simulated to a reasonable degree of accuracy (but mostly to be a bit pretty and to break up monotony in galaxies). 
 - Different star populations are found throughout galaxies. In spirals, the bulge/bar has lower mass, redder stars. The leading edges of spirals have high mass blue stars, with lower mass redder stars trailing the leading edges. There are also a population of even lower mass stars (on average) not associated with spirals. Elliptical galaxies are characterised by many more, lower mass red stars.
 - Rotation curves are simulated accurately according to newtonian physics, with black holes and dark matter (via the Navarro-Frenk-White profile) influencing rotation velocities. 
 
![](/MiscGithubImages/localradialvel.jpg)
### Black Holes
<img align="right" height="200" src="/MiscGithubImages/radiolobetypes.jpg">  

 - Massive black holes (usually on the order of 50-500 solar masses) are generated at the center of all/most galaxies with the mass dependent on the galaxy mass as a whole. 
 - Black holes all have a luminosity on the order of the eddington luminosity for a black hole of that mass. 
 - Mock 'Radio Lobes' from black hole accretion shoot out from the 'poles' of the galaxy (which assumes that the accretion disk is parallel with the plane of the galaxy). Spiral galaxies display Fanaroff-Riley Type I lobes, while elliptical galaxies shown FRII lobes. At the moment, there is no use for this other than qualitative means. 
 - Black holes have a cluster of ~20 massive stars around them by default, with random velocity directions. In the unlikely case you can actually see the black hole in an output image, it shows up as a dot with an aqua-ish colour (according to the blackbody colours link in the credits). 
### Galaxy Clusters
<p align="middle"><img src="/MiscGithubImages/Cluster1.jpg" height="200" /><img src="/MiscGithubImages/galaxradialvels.jpg" height="200" /></p>  

 - Galaxy clusters generate at least one galaxy according to an exponential distribution with a mean of 8 galaxies. Clusters with 5 or more galaxies will have an elliptical galaxy at their center, and clusters with 10 or more galaxies will have a cD galaxy in their center.
   - Elliptical galaxies are much more common close to the center of clusters. Conversely, spirals are more likely on the outer edges of clusters. 
 - Using a similar approach to that of galaxy rotation curves, galaxies have a rotation velocity about the cluster center in a random direction (much similar to the method used for elliptical galaxies!)

### The Universe as a whole
<img align="right" height="200" src="/MiscGithubImages/Hubble Diagram.png">  

 - Hubble recession is modelled, with more distant galaxies receeding further away. The hubble constant itself is chosen as a value between 1000 and 5000km/s/Mpc, with a random sign. 
 - Radial velocities of resolved stars take into account hubble recession, galaxy movement within clusters, and star movement within galaxies. 
 - In output visual images of the universe (see the top of this readme!), stars have diffraction spikes according to how bright they appear to the observer. Why? Because this is pretty and I like it. 
 - Type Ia supernova are randomly exploded across ~55 or so galaxies in the universe, with at least two of them being in the local cluster of galaxies so that the user may more easily find the intrinsic brightness. 
 - Homogeneous and inhomogeneous universes (inhomo by default) are able to be generated. Homogeneous has spherically-uniformly distributed galaxies, while inhomogeneous has exponentially increasing cluster counts up to a threshold of 30kpc (this is where normal galaxies are no longer generated and the 'distant galaxies' are now generated), where cluster count then exponentially decreases towards the radius of the universe. 


## To-Do:
- [ ] Make this damn README.md prettier!!
- [ ] Include a 'how to use' section in this readme (thanks saskia!)
- [ ] Fill out many docstrings and comment all code to a degree readable by other users.
- [x] Generate individual datasets inside of a "Datasets" folder. 
- [x] Randomly determine hubble constant (and update the readme after doing so). 
- [ ] Set it so that some galaxies don't have a black hole and/or dark matter, with probability depending on galaxy type. 
- [ ] Output some luminosity data about black holes. Maybe under x-ray data?
- [ ] Make black hole proportion of eddington luminosity depend on same manner of the host galaxy. 
- [ ] Make radio lobe brightness depend on black hole luminosity.
- [ ] Make brightness of black holes depend on angle of the galaxy to the viewer? As well as mass and random activity (possibly depending on galaxy type)

## Credits/Acknowledgements
 - Saskia for providing a mental amount of help with regards to Python programming. I've learnt a lot! Also many many sanity checks and FEEDBACK. Mamma mia. 
 - Ciaran for helping with a bit of math (linear algebra is hard) and some astro sanity checks here and there. 
 - The `blackbodycolours.txt` file was supplied free of charge by Mitchell Charity ([email](mailto:mcharity@lcs.mit.edu)) from [What colour is a blackbody?](http://www.vendian.org/mncharity/dir3/blackbody/) (Version 2001-Jun-22)
 - The astrophysics team at the University of Queensland for providing inspiration for this project in the form of their simulation data! 
