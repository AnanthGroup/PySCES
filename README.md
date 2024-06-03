# PySCES
PySCES is a highly parallelized code for ab initio nonadiabatic molecular dynamics using linearized semiclassical initial value representation (LSC-IVR). As of now, on-the-fly updates of electronic structure variables can be performed either by GPU-assisted electronic structure package, TeraChem, or, as its original implementation was structured, GAMESS. The code is intended for computing electronic population correlation functions through one of three population estimators: Wigner, semiclassical, or spin mapping population estimator. See the following references for the original implementation and the application of the code:

- K. Miyazaki and N. Ananth. "Nonadiabatic simulations of photoisomerization and dissociation in ethylene using ab initio classical trajectories," J. Chem. Phys. 159, 124110 (2023)
- https://doi.org/10.1063/5.0163371

References for the most recent implementation will be added as they become ready.

Created by Christopher Myers, Ken Miyazaki, Thomas Werner Trepl

Disclaimer: The code contained in this package has been written, edited, and used by members of the Ananth group at Cornell University and the Isborn group at the University of California, Merced. It has not been formally reviewed, nor published, nor are there copyrights. Bugs and errors may be present.
