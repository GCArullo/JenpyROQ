# JenpyROQ
This repository implements an extended, modularised and streamlined version of the PyROQ code (see below for a detailed list of changes), forked in March 2022 from its [original repository](https://github.com/qihongcat/PyROQ) and developed at the Theoretisch-Physikalisches Institut of the Friedrich-Schiller-Universität Jena. 

Older history is available in [this fork](https://github.com/bernuzzi/PyROQ/tree/master/PyROQ).
Please cite the [PyROQ paper](https://arxiv.org/abs/2009.13812) and the COMING SOON paper if you use this code in your research. 

# Installation and usage

Starting from the `JenpyROQ` location, 
the package can be installed using the command:

    python setup.py install

Once  `JenpyROQ` is  installed, it is possible to construct an ROQ approximant through a configuration file and running the main routine of the package:

    python -m JenpyROQ --config-file config.ini

The user can see the full list of options at:

    python -m JenpyROQ --help

A simple example can be run by:

    python -m JenpyROQ --config-file config_files/Test_configs/test_config_IMRPv2.ini

Other examples are available in the `config_files` directory, see the relative [README file](https://github.com/bernuzzi/PyROQ/blob/master/config_files/Test_configs/README.md).

## MPI parallelisation

Parallelisation options are described under the `[Parallel]` section of the help message.
For MPI-based parallelisation, the run command should additionally be modified as follows:
    
    mpiexec -n NTASKS python -m JenpyROQ --config-file config_files/Test_configs/test_config_IMRPv2.ini
    
Where `NTASKS` corresponds to the requested number of parallel tasks. 
Moreover, the config file should specify the related flag `parallel=2`  and `n-processes` should correspond to NTASKS.  


# Output

The run directory will automatically contain a copy of the configuration file, git information and the screen output, stored under `JenpyROQ.log`.

Preselection basis and related parameters, together with the enriched basis, its related parameters, the basis interpolant and empirical nodes are stored at each step of the enrichment loop under the `ROQ_data` directory.

Basis specifications needed to interface with parameter estimation codes are available under `ROQ_data/ROQ_metadata.txt`.

Several diagnostic plots (basis parameters, frequency nodes, outliers and error evolution, a single test waveform comparison and validation tests) are stored under the `Plots` directory.

# Algorithm and code description

The algorithm implements what described in the [PyROQ paper](https://arxiv.org/abs/2009.13812), but extended to allow for a more flexible pre-basis construction (with user-defined inputs), an arbitrary number of enrichment cycles with arbitrary and variable tolerance thresholds.

The code also speeds up the execution by parallelising wherever possible, implements a more strict parameters and input/output management, a largely increased use of code modularisation, defines generic waveform structures capable of interfacing modern python-based models, including machine learning-based ones.

Additional models supported compared to the original version (which supports any waveform approximant implemented in `LALSimulation`) are: 

   * [`TEOBResumS`](https://bitbucket.org/eob_ihes/teobresums/src/master/README.md), a faithful semi-analytical model for compact objects on generic orbits;  
   * [`MLGW-BNS`](https://pypi.org/project/mlgw-bns/), an efficient machine learning version of the frequency-domain [`TEOBResumSPA`](https://arxiv.org/abs/2012.00027) model; 
   * [`NRPMw`](https://arxiv.org/abs/2205.09112), post-merger (including their hybridisation with EOB-inspirals) binary neutron star models, interfaced through [`bajes`](https://github.com/matteobreschi/bajes).

## Adding a new parameter

The parameters structure logic is: a default set of parameters (a combination of the masses) is always included in the training range. For each new set of parameters added (e.g. spins, or tides) a flag will control the activation of such set. Default values of the new set of parameters should be declared, together with activation rules mediating the usage of the new parameter and the interaction with other options.

Specifically, if you intend to add a new set of parameters (following e.g. the appearance of the `'tides'` flag and `'lambda1'` parameter):

1. Declare the flag activating the new set of parameters [here](https://github.com/GCArullo/JenpyROQ/blob/2423ef8fffe14b2c996b3a6bfd36743b929f1bc4/JenpyROQ/initialise.py#L262), and document the flag [here](https://github.com/GCArullo/JenpyROQ/blob/2423ef8fffe14b2c996b3a6bfd36743b929f1bc4/JenpyROQ/initialise.py#LL58C16-L58C21). Flag conventions are documented [here](https://github.com/GCArullo/JenpyROQ/blob/2423ef8fffe14b2c996b3a6bfd36743b929f1bc4/JenpyROQ/initialise.py#L29);

2. Document the new parameters [here](https://github.com/GCArullo/JenpyROQ/blob/2423ef8fffe14b2c996b3a6bfd36743b929f1bc4/JenpyROQ/initialise.py#L119), and declare their default and test values [here](https://github.com/GCArullo/JenpyROQ/blob/2423ef8fffe14b2c996b3a6bfd36743b929f1bc4/JenpyROQ/initialise.py#L192);

3. Declare the activation rule of the new parameter [here](https://github.com/GCArullo/JenpyROQ/blob/2423ef8fffe14b2c996b3a6bfd36743b929f1bc4/JenpyROQ/initialise.py#L156), together with possible interaction with other options, e.g. [here](https://github.com/GCArullo/JenpyROQ/blob/2423ef8fffe14b2c996b3a6bfd36743b929f1bc4/JenpyROQ/initialise.py#L138); 

4. Regulate their usage in the waveform call e.g. [here](https://github.com/GCArullo/JenpyROQ/blob/2423ef8fffe14b2c996b3a6bfd36743b929f1bc4/JenpyROQ/waveform_wrappers.py#L316).

# Dependencies

The package depends on standard Python libraries, except for: `numpy` for numeric computation, `h5py` for data storing and `matplotlib` for plotting. Moreover, if MPI-based parallelisation is requested, the package has an additional dependency on `mpi4py`.

# Development history

SB (sebastiano.bernuzzi@uni-jena.de) 03/2022:
   * Forked PyROQ version 0.1.26
   * Added support for [TEOBResumS GIOTTO](https://bitbucket.org/eob_ihes/teobresums/src/master/) and MLW-BNS
   * Refactored code
     - Introduced JenpyROQ class
     - Simplified code/reduced duplication
     - Added waveform wrapper classes
     - Changed parameter management

GC (gregorio.carullo@uni-jena.de) 05/2022:
  * Debugged and simplified `refactored` branch.
  * Switched to config file usage.
  * Implemented algorithm as described in PyROQ paper: pre-selection loop and subsequent enrichment cycles.
  * Allow user to determine an arbitrary number of enrichment cycles.
  * Allow user to determine variable and arbitrary tolerance thresholds.
  * (Almost) maximally streamline code and move logically separated functions to specific files.
  * Parallelise linear and quadratic, add more parallelisation steps where possible.
  * Improve post-processing and input/output storage (git info, config file, stdout/stderr).
  
MB (matteo.breschi@uni-jena.de) 05/2022:
  
  * Introduce logger
  * Implement MPI-based parallelisation and unify pool usage
  * Extend setup.py, improve packaging and include main functionalities
  
GC (gregorio.carullo@uni-jena.de) 06/2022:
  
  * Introduce inversion stability checks and nodes repetition checks to flag ill-conditioned execution or algorithm failures;
  * Enforce controlled parameters set logic (new flag for each set of parameters);
  * Add support for several starting basis options;
  * Add large number of example config files;

GC (gregorio.carullo@uni-jena.de) and MB (matteo.breschi@uni-jena.de) 08/2022:

  * Add support for NRPMw and EOB-NRPMw models.
