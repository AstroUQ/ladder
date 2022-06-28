# Simulated-Universe
## About
I am intending to emulate the "Distance Ladder" universe simulation that was a part of the UQ course PHYS3080, potentially adding a few bells and whistles along the way. 

## To-Do:
 - Stars:
    - Simulate variability in some regions on the HR diagram. 
 - Galaxies:
	- Change dark matter properties based on galaxy type (in rotation velocity function)
 - Black Holes:
    - Make brightness depend on angle of the galaxy to the viewer? As well as mass and random activity (possibly depending on galaxy type)
 - Galaxy Clusters:
    - Rotation/Dispersion curves for galaxies within a cluster
 - Type Ia supernovae as a standard candle
    - Need >~55 across the whole universe
        - Randomly select clusters to host them?
        - Need at least two close enough to the observer in order for them to evaluate intrinsic brightness. 
 - Generate a new type of galaxy (very distant) that is not comprised of stars (maybe use an image with random rotation?)
    - Needs size (arcseconds), luminosity data, radial velocity and position. 
 - Data output/creation:
    - Generate parallax angles for all stars
	- Divide luminosities by distance squared
