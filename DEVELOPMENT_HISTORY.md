# Development history

SB (sebastiano.bernuzzi@uni-jena.de) 03/2022:

* Forked PyROQ version 0.1.26.
* Added support for [TEOBResumS GIOTTO](https://bitbucket.org/eob_ihes/teobresums/src/master/) and MLW-BNS.
* Refactored code:
  * Introduced the `JenpyROQ` class.
  * Simplified code and reduced duplication.
  * Added waveform wrapper classes.
  * Changed parameter management.

GC (gregorio.carullo@uni-jena.de) 05/2022:

* Debugged and simplified the `refactored` branch.
* Switched to config file usage.
* Implemented the algorithm as described in the PyROQ paper: pre-selection loop and subsequent enrichment cycles.
* Allowed users to determine an arbitrary number of enrichment cycles.
* Allowed users to determine variable and arbitrary tolerance thresholds.
* Streamlined code and moved logically separated functions to specific files.
* Parallelised linear and quadratic construction and added more parallelisation steps where possible.
* Improved post-processing and input/output storage: git information, config file, stdout and stderr.

MB (matteo.breschi@uni-jena.de) 05/2022:

* Introduced logger.
* Implemented MPI-based parallelisation and unified pool usage.
* Extended `setup.py`, improved packaging and included main functionalities.

GC (gregorio.carullo@uni-jena.de) 06/2022:

* Introduced inversion stability checks and node repetition checks to flag ill-conditioned execution or algorithm failures.
* Enforced controlled parameter-set logic with a new flag for each set of parameters.
* Added support for several starting basis options.
* Added a large number of example config files.

GC (gregorio.carullo@uni-jena.de) and MB (matteo.breschi@uni-jena.de) 08/2022:

* Added support for NRPMw and EOB-NRPMw models.
