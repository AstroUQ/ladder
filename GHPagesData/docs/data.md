# What's inside the data...

When opening the dataset folder, you'll be greeted by a few files all of which are relevant to working out properties of the universe within.

### `Star_Data.csv`
Arguably the most important of the files are the star data files. Each _**resolved**_ star in the universe has a line of data in these files. Measured is the stars position, its monochromatic (meaning one colour) luminosity for three wavelengths (using a highly sensitive instrument), parallax angle, radial velocity and a light curve (if the star is characterised by a variable luminosity). 

 Measurement | Unit | Explanation
 --- | --- | --- 
 X | degrees | Horizontal position of the star in the image
 Y | degrees | Vertical position of the star in the image
 BlueF | W/nm/m^2^ | Monochromatic luminosity of the star at 440nm
 GreenF | W/nm/m^2^ | Monochromatic luminosity of the star at 500nm
 RedF | W/nm/m^2^ | Monochromatic luminosity of the star at 700nm
 Parallax | arcseconds | Maximum parallax angle of the star across 1 year of observations
 RadialVelocity | km/s | Line-of-sight motion of the star (positive implies motion away)
 Variable? | N/A | 1 if the star shows variability in its luminosity, 0 otherwise.

### `Variable Star Data`
Some stars in the universe will have a variable luminosity across time, and so this variability was measured if applicable and saved in a separate file for each star in the format `{starname}.csv`. The luminosity of the star was measured once every hour for 120 hours, and data is given in terms of a proportion of its average luminosity. 

 Measurement | Unit | Explanation
 --- | --- | --- 
 Time | hours | The time (since starting) of the measurement
 NormalisedFlux | N/A | Proportion of baseline (average) luminosity

### `Distant_Galaxy_Data.csv`
Due to their small size in the sky, distant galaxies could be measured using the same instruments as those used to measure star properties. 

 Measurement | Unit | Explanation
 --- | --- | --- 
 X | degrees | Horizontal position of the galaxy
 Y | degrees | Vertical position of the galaxy
 BlueF | W/nm/m^2^ | Monochromatic luminosity of the galaxy at 440nm
 GreenF | W/nm/m^2^ | Monochromatic luminosity of the galaxy at 500nm
 RedF | W/nm/m^2^ | Monochromatic luminosity of the galaxy at 700nm
 Size | arcseconds | The width in the sky subtended by the galaxy
 RadialVelocity | km/s | Line-of-sight motion of the galaxy (positive implies motion away)

### `Flash_Data.csv`
Across the period of observation (1 year), several extremely bright flashes were observed. The number of X-Ray photons observed were recorded using the all-sky monitor.

 Measurement | Unit | Explanation
 --- | --- | --- 
 Direction | N/A | The image face that the X/Y coordinates relate to
 X | degrees | Rough horizontal position of the bright flash on the specified image face
 Y | degrees | Rough vertical position of the bright flash on the specified image face
 Photon-Count | N/A | Number of X-Ray photons detected from the source