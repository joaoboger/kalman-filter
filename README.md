# kalman-filter
The intent is to learn how Kalman Filter works at CMS Trigger.

-To plot data with concentric circles: open "ConcentricKalman.C" and look for "file.open("/path/to/DataFile.txt", ios::in);" and insert the path to the data file. 
It reads line by line the data file using the data as x-coordinate, y-coordinate, x-error, y,error throughout the whole file, so make sure data is organized this way.

-There are made-up data files in this repo to test plots and new additions to code here named as "XParticles_errorPhiYpZ.txt" where X stands for the number of particles, and the corresponding associated error Y.Z with the data at each point in the Phi coordinate(Polar coordinates) .

-The file "2dCurvesKalmanFilter.py" allows you to use Kalman Filter with the desired equations of motion through the function "updatePosition(r, phi, vr, vphi, dt, j, q)" where **r**,**phi** stands for the initial position in polar coordinates; **vr**, **vphi** the respective velocities, **dt** the time steps of the propagation model, **j** which detector are you considering to detect the state and **q** the particle charge in elementar units.

# RK-4
-To apply RK-4 on the equations of motion I'm using the function 'odeint' from 'scipy.integrate' library.
