# Semiogram: a Visual Tool for Gait Quantification in Routine Neurological Follow-Up

## Reference 

IPOL link: 
Date: 
Author mail: cyril.voisard@gmail.com

## Context

This repository provides the code for calculating the parameters of the Semiogram, a tool for quantifying gait using inertial measurement units. In summary, the Semiogram can be defined as a **performance bar** for an individual according to different criteria divided into 7 axes and 1 coloured global axis (speed). 
The values for each of these criteria are compared for the individual with the average for those in the healthy age group. They are therefore given as a percentage of the standard deviation.

The first stage of scientific validation of this tool was published in 2023 in the journal Frontiers [Voisard et al, Innovative multidimensional gait evaluation using IMU in multiple sclerosis: introducing the semiogram, Frontiers in Neurology, 2023, https://doi.org/10.3389/fneur.2023.1237162]. 
This article explains each of the 17 parameters and 8 semiological criteria of gait and explores its clinical application in patients with multiple sclerosis. 

An article detailing each of the mathematical formulas used to calculate the parameters and the associated lines of code has also been published in IPOL : 
[Voisard et al., Semiogram: a Visual Tool for Gait Quantification in Routine Neurological Follow-Up, IPOL, 2024, XXXXX].


## Demo and accessibility 

The demo applying the calculation and representation of the semiogram to data supplied in the correct format is available here: https://www.ipol.im/pub/art/XXXX.
The format of the input data and the output figures and tables are provided in the attached article. 


## Examples of use

3 datasets are provided in the folder "example_datasets" consisting of 2 gait signals collected using an inertial measurement units (MTw Awinda XSens, in the lower back at the level of the fifth lumbar vertebra), with a gait events segmentation. The data was obtained from a sample of subjects who followed the described protocol: standing still, walking 10 meters, turning around, walking back, and stopping.
The data output by the algorithm is provided in the same files. 

## Citations 

Please cite these articles whenever you want to make a reference to this demo:
- [Voisard et al, Innovative multidimensional gait evaluation using IMU in multiple sclerosis: introducing the semiogram, Frontiers in Neurology, 2023, https://doi.org/10.3389/fneur.2023.1237162]
- [Voisard et al., Semiogram: a Visual Tool for Gait Quantification in Routine Neurological Follow-Up, IPOL, 2024, XXXXX]
