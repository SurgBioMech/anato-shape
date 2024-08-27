# anato-shape
A library of functions and notebooks to perform curvature calculations on 3D surface meshes of complex anatomies.

Required libraries:
* sklearn
* open3d
* trimesh
* plotly (just for viz functions)

# Data file organization structure the code is current written to work with: 
Parent Directory: `Z://aorta/aortas`
+
Cohort Groups e.g., `KK` or `KY` 
+
Patient ID & Scan Number e.g., `KK1` and `KK1_1` (*"_1" refers to the first scan, "_2" is the second scan, and so on)
+
Required file holding subfolder: `mesh`
=
Example file structure needed: `Z://aorta//aortas//KK//KK1//KK1_1//mesh`
