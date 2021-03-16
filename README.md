# kalman-filter
The intent is to learn how Kalman Filter works at CMS Trigger.

-To plot data with concentric circles: open "ConcentricKalman.C" and look for "file.open("/path/to/DataFile.txt", ios::in);" and insert the path to the data file. 
It reads line by line the data file using the data as x-coordinate, y-coordinate, x-error, y,error throughout the whole file, so make sure data is organized this way.

-There are made-up data files in this repo to test plots and new additions to code here named as "XParticles_errorPhiYpZ.txt" where X stands for the number of particles, and the corresponding error associated with the data at each point in the Phi coordinate(Polar coordinates) Y.Z .
