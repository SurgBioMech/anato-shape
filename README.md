# anato-shape
A library of functions and notebooks to perform curvature calculations on 3D surface meshes of complex anatomies.

anato-curvature-notebook.ipynb is a Jupyter notebook used to calculate per-vertex curvatures values using mesh input files (`.mat`) and return the mesh and values in a compressed file (`.parquet`). 

anato-mesh-notebook.ipynb then takes those `.parquet` file meshes and performs the partitioning and higher-level statistical curvature analysis and returns the results for a batch of unique anatomies is a singular summary table. 



Data file organization structure the code is currently written to work with: 

Parent Directory: `Z://aorta/aortas`

Cohort Groups e.g., `KK` or `KY` 

Patient ID & Scan Number e.g., `KK1` and `KK1_1`

Required file holding subfolder: `mesh`

Example file structure needed: `Z://aorta//aortas//KK//KK1//KK1_1//mesh`



Required libraries:
* sklearn
* open3d
* trimesh
* plotly (just for viz functions)
