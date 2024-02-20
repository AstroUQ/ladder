---
title: Home
description: Introduction Page
---
# PHYS3080: Distance Ladder Project

<p align="middle"><img src="assets/solar system.png" style="width:15%"/><img src="assets/local stars.png" style="width:15%"/><img src="assets/parent galaxy.png" style="width:15%"/><img src="assets/local galaxies.png" style="width:15%"/><img src="assets/distant clusters.png" style="width:15%"/><img src="assets/observable universe.png" style="width:15%"/></p>

## Context

The planet of New Earth is engulfed in one of the thickest atmospheres in the Universe. Although New Earthlings are highly intelligent, their concept of astronomy is relatively new. When their first rockets penetrated the clouds and snapped pictures of the sky above, they discovered their planet was orbiting a star (along with other planets) and the sky was filled with other sources of light (other stars?).

Many rockets later, the New Earthlings have gained a good understanding of their own solar system. Very little, however, is known about the rest of their Universe. It is known that there are many dots of light, and some small fuzzy glowing patches, which do not seem to move across the sky, and which are probably not part of the solar system. Spectra have been obtained of a few of these dots and fuzzy patches - the spectra look like black body curves with absorption-lines superimposed. They seem to have a range of temperatures (typically a few thousand K), and to be made of elements such as Hydrogen and Carbon, similar to New Earth's parent Sun.

Over the last few years, you have helped build the planet's most ambitious (and expensive) space-probe yet: the Pimbblet Picture Producer (named after an Antipodean scientist with a rare surname who liked thinking about BIG questions). This remarkable satellite contained a number of instruments:

- Six wide-field cameras, each of which repeatedly took pictures of 1/6 of the sky. Together, they measured the brightness and position of each object in the sky repeatedly. Variable objects were automatically identified and light curves generated for them.
- A high resolution, narrow-field camera, which took more detailed close-up images of smaller regions. It took two images of each part of the sky, spaced by six months. It detects every object brighter than 10<sup>-22</sup> W m<sup>-2</sup> nm<sup>-1</sup> in its field of view. Its six month spacing also allowed it to measure the parallax of each source, as New Earth moves around its Sun (New Earth is 1 AU from its Sun, and orbits it once a year).
- A spectrometer, which measured the wavelengths of the various absorption-lines, and hence used the Doppler effect to determine the radial velocity of each object detected by the imaging cameras.
- An X-ray all-sky monitor which had 1m<sup>2</sup> detector area, which scanned the sky looking for any X-rays coming from space.

All these instruments were programmed to not take data whenever anything in New Earth's solar system was in their field of view. This was to prevent the cameras being damaged by such bright targets. As a result, you will not see anything from New Earth's solar system in these data.

The space-probe has just landed after its one year mission, and the data tapes eagerly retrieved by your team. You have just started to analyse these exciting and unprecedented data.


## Sitemap
On the [Introduction page](tutorials/introduction.md) you'll find all of the physics that is actually emulated in this simulation, as well as some important information about the simulation on the [Data Structure page](data.md). Check your group identification and dataset on the [datasets page](datasets.md), and the data structure on the [data page](data.md).

## Hints

Your job, in general terms, is to work out as much as you can about New Earth's Universe. You can assume that basic physical constants like the speed of light or the constant of gravity are the same as in our universe. However, the more complicated physics like star formation, stellar and galaxy evolution are probably different. To what extent they differ is something that you will have to work out. The more you work out, the better your score in this assignment will be. Here are some some unsorted, non-exhaustive hints and pointers about the exercise:

There are no marks for just coming up with lists of numbers -- we are interested in the physics of your given Universe.

- Does your Universe contain galaxies? Galaxies with disks? How big are the disks (radius and thickness)? Do they have dark matter? Which galaxy does New Earth reside in? Can you produce an H-R diagram for New Earth's galaxy?
- Is the cosmological principle obeyed (i.e. homogeneity and isotropy?)
- What are the X-ray flashes all about? Why do they occur and how frequently?
- How massive are galaxies in your Universe? How many stars per galaxy (typically)? What are the temperatures and radii of these stars? What's their mass to light ratio?
- How far away are the galaxies in your Universe? More importantly: can you construct a convincing distance ladder?
- Can you determine if you have any groups or clusters of galaxies? What are their populations and stellar populations like? What is the ratio of total cluster mass to luminous cluster mass? Do the galaxies contain black holes?
- Compute Hubble's constant. How old is your Universe?
- Only for those of you feeling very ambitious!
	- How did your Universe begin and what is its future? Is there life elsewhere in your galaxy or Universe?

Much like investigating the real Universe, we don't expect you to address ALL of the above questions in a superbly detailed manner. You must pick and choose which questions or aspects of this project you want to concentrate on.

However: We suggest that your starting point be constructing a distance ladder. You will get at least a 5 for an appropriately well written report *if* it contains a reasonably accurate determination of H0, with errorbars. This is your first objective! 

## Deadlines and Grading

Please note that this is not an assignment that you can leave until the last possible moment. We will be spending a number of sessions in the computer lab for you to investigate the complex dataset, but you should also be continuing with this exercise in your own time. You have to do the following:

- Read the marking scheme [here](assets/distance_ladder_report_rubric.pdf).
- Work in your groups to help each other solve the important parts of the problem.
- Document your contribution to the group work on the group discussion board. You must make at least one significant posting at least a week before the final deadline.
- Submit a full written report. This is your own report: you can use results from other group members (if acknowledged) but the writing must all be your own. It needs an abstract section that gives clear statements of your answers; see the marking scheme for more details. You may place detailed calculations in an appendix so as to not clutter up the body of the text. Please do not include code or lengthy data tables, even in an appendix. We will require an electronic copy of your assignment to be uploaded via Turnitin (accessed through Blackboard).
- In your project work and in the reports, concentrate on getting physical information out of the data. Avoid speculations that cannot be backed up by actual data. Also avoid drawing parallels to our universe. To the astronomers in New Universe, our universe is not available for comparison. Any physics you could know from the lab is safe, but any astrophysics could be completely different. 
- Deadline for the written component? See Course Profile. Note the time deadline for electronic submission is strict and exact!

