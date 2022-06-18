# Simulated-Universe
## About
I am intending to emulate the "Distance Ladder" universe simulation that was a part of the UQ course PHYS3080, potentially adding a few bells and whistles along the way. 

## To-Do:
 - Stars:
    - Generate narrow band luminosity data (currently have bolometric luminosity and temperature). 
 - Create standard galaxy populations and radii for galaxy types. 
 - Galaxy Rotation Curves
    - Curves given black hole and no black hole
    - Curves given dark matter and no dark matter (smaller galaxies might not have dark matter?)
    - Star velocities given their distance from galaxy center
        - Radial velocity that the observer sees (line of sight motion)
 - Create Black Hole Class
    - Mass dependent on central bulge mass? Or maybe galaxy mass
    - Make brightness depend on angle of the galaxy to the viewer? As well as mass and random activity (possibly depending on galaxy type)
    - Somehow make radio emission contours for elliptical galaxies?
 - Galaxy Clusters
    - E0 Galaxies in the center of clusters
        - Higher chance of ellipticals closer to the center of the cluster
    - Rotation/Dispersion curves for galaxies within a cluster
 - Type Ia supernovae as a standard candle
    - Need ~55 across the whole universe
        - Randomly select clusters to host them?
        - Need at least two close enough to the observer in order for them to evaluate intrinsic brightness. 
 - Generate a new type of galaxy (very distant) that is not comprised of stars (maybe use an image with random rotation?)
    - Needs size (arcseconds), luminosity data, radial velocity and position. 
 - Misc
    - Generate parallax angles for all stars
