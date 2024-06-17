# PySCES
PySCES, **Py**thon code for Linearized **S**emi-**C**lassical Dynamics with On-the-fly **E**lectronic **S**tructure, is a highly parallelized code for ab initio nonadiabatic molecular dynamics using linearized semiclassical initial value representation (LSC-IVR). As of now, on-the-fly updates of electronic structure variables can be performed either by GPU-assisted electronic structure package, TeraChem, or, as its original implementation was structured, GAMESS. The code is intended for computing electronic population correlation functions through one of three population estimators: Wigner, semiclassical, or spin mapping population estimator. See the following references for the implementation and the application of the code:

- C. A. Myers, K. Miyazaki, T. Trepl, C. M. Isborn, and N Ananth. "GPU-Accelerated On-the-fly Nonadiabatic Semiclassical Dynamics," J. Chem. Phys., XXXX, XXXXXX, (2024)

- K. Miyazaki and N. Ananth. "Nonadiabatic simulations of photoisomerization and dissociation in ethylene using ab initio classical trajectories," J. Chem. Phys. 159, 124110 (2023), https://doi.org/10.1063/5.0163371

## Installation
The easiest way to get PySCES is to clone the repository from github. As the code is still in active development, this makes it convenient to get the latest changes as they are uploaded. PySCES can be installed with pip, and we recommend you make a new python environment with Conda prior to the installation:
```
    conda create -n pysces python
    conda activate pysces
    cd /dir/where/you/git/cloned/
    pip install -e .
```
The `-e` will tell pip not to copy over the code itself into your python environment. To get the latest updates, just go back to the location you cloned the repository, run a `git pull`, and your installation will also be updated. You do not need to run a `pip install` again.

## Running PySCES
After installation, the command `pysces` will be registered with your conda environment and is used to initialize and run the simulation. After running this command, PySCES will look for an input file `input_simulation_local.py` in the current directory. This file is formatted in python and sets the primary variables that control the simulation.  

Each input will usually contain the following information:
* Number of atoms in the molecule system.
* Number of electronic states to be simulated.
* Temperature to sample nuclear Wigner distributions from.
* The electronic state initially photoexcited.

Examples of this input file can be found in the `examples` directory of the main repository and can be used as basic templates for running simulations. The variable names used to control the simulation, including the settings listed above, are described in the manual.
