# PySCES
PySCES, **Py**thon code for Linearized **S**emi-**C**lassical Dynamics with On-the-fly **E**lectronic **S**tructure, is a highly parallelized code for ab initio nonadiabatic molecular dynamics using linearized semiclassical initial value representation (LSC-IVR). As of now, on-the-fly updates of electronic structure variables can be performed either by GPU-assisted electronic structure package, TeraChem, or, as its original implementation was structured, GAMESS. The code is intended for computing electronic population correlation functions through one of three population estimators: Wigner, semiclassical, or spin mapping population estimator. See the following references for the original implementation and the application of the code:

- K. Miyazaki and N. Ananth. "Nonadiabatic simulations of photoisomerization and dissociation in ethylene using ab initio classical trajectories," J. Chem. Phys. 159, 124110 (2023)
  - https://doi.org/10.1063/5.0163371

References for the most recent implementation will be added as they become available.

Created by Christopher Myers, Ken Miyazaki, and Thomas Werner Trepl.

Disclaimer: The code contained in this package has been written, edited, and used by members of the Ananth group at Cornell University and the Isborn group at the University of California, Merced. It has not been formally reviewed, nor published, nor are there copyrights. Bugs and errors may be present.

## Installation

The easiest way to get PySCES is to clone the repository from github. As the code is still in active development, this makes it convenient to get the latest changes as they are uploaded. PySCES can be installed with pip, and we recommend you make a new python environment with Conda prior to the installation:
```
    conda create -n pysces python
    conda activate pysces
    cd /dir/where/you/git/cloned/
    pip install -e .
```
The `-e` flag will tell pip not to copy over the code itself into your python environment. To get the latest updates, just go back to the location you cloned the repository, run a \texttt{git pull}, and your installation will also be updated. You do not need to run a `pip install` again.

## Usage
For further details, including the various options that can control PySCES, please read the Manual in PDF form within the repository. 
