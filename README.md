# AI-LSC-IVR
AI-LSC-IVR is a code written in Python3 to perform ab initio nonadiabatic dynamics using linearized semiclassical initial value representation (LSC-IVR). The current version of the code is interfaced with an electronic structure package GAMESS and computes an electronic population correlation function using Wigner, semiclassical, or spin mapping population estimator. See the following references for the implementation and the application of the code:

- C. A. Myers, K. Miyazaki, T. Trepl, C. M. Isborn, and N Ananth. "GPU-Accelerated On-the-fly Nonadiabatic Semiclassical Dynamics," \textit{J. Chem. Phys.}, XXXX, XXXXXX, (2024)

- K. Miyazaki and N. Ananth. "Nonadiabatic simulations of photoisomerization and dissociation in ethylene using ab initio classical trajectories," J. Chem. Phys. 159, 124110 (2023), https://doi.org/10.1063/5.0163371

PySCES was initially written by Ken Miyazaki while at Cornell University. Christopher Myers extended the code to call TeraChem as the electronic structure driver at the University of California, Merced. The authors would also like to thank Thomas Trepl at University of Bayreuth, Germany, for his contributions to the code, including procedures for correcting nonadiabatic coupling sign-flips. 

Disclaimer: The code contained in this package has been written, edited, and used by members of the Ananth group. It has not been formally reviewed, nor published, nor are there copyrights. Bugs and errors may be present.


