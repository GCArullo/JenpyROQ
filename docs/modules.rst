Module Reference
----------------

This page is a source-oriented map rather than an autodoc page. The package
imports optional waveform libraries at module import time, so importing every
module during a documentation build can fail on machines that only have the
core documentation dependencies installed.

Command Entry Point
~~~~~~~~~~~~~~~~~~~

``JenpyROQ.__main__`` handles:

* parsing ``--config-file``;
* creating output directories;
* setting the ``JenpyROQ`` logger;
* selecting serial, multiprocessing or MPI execution;
* setting random seeds;
* looping over linear and quadratic run types;
* saving metadata and launching diagnostics.

Configuration Parser
~~~~~~~~~~~~~~~~~~~~

``JenpyROQ.initialise`` handles:

* the help text shown by ``python -m JenpyROQ --help``;
* default configuration dictionaries;
* type conversion from INI strings;
* list parsing for enrichment-cycle options;
* activation and skipping of sampled parameters;
* training-range and test-value parsing;
* git-state recording.

ROQ Core
~~~~~~~~

``JenpyROQ.jenpyroq.JenpyROQ`` owns the core state:

* active waveform approximant;
* parameter index maps;
* frequency grid;
* basis tolerances and training-cycle controls;
* parameter transforms from ``mc,q`` or ``m,q`` to ``m1,m2``;
* waveform generation through the selected wrapper;
* pre-selection and enrichment loops;
* empirical interpolation nodes and basis interpolants.

Waveform Wrappers
~~~~~~~~~~~~~~~~~

``JenpyROQ.waveform_wrappers`` defines the wrapper registry ``WfWrapper`` and
the non-LAL approximant names. The registered wrapper class depends on optional
imports:

* ``LALWf`` for LALSimulation approximants;
* ``WfTEOBResumS`` for ``teobresums-giotto``;
* ``WfMLGW`` for ``mlgw-bns-standalone``;
* ``WfBajes`` for bajes and NRPMw-family approximants.

Linear Algebra
~~~~~~~~~~~~~~

``JenpyROQ.linear_algebra`` exposes:

.. list-table::
   :header-rows: 1
   :widths: 30 54

   * - Function
     - Purpose
   * - ``scalar_product(vec1, vec2, df, weights=1.)``
     - Computes ``4 * df * real(vdot(vec1, vec2))`` after optional weighting.
   * - ``normalise_vector(vec, df)``
     - Divides a vector by the square root of its scalar product.
   * - ``projection(u, v)``
     - Computes the projection helper used during basis construction.
   * - ``gram_schmidt(basis, vec, df)``
     - Iteratively subtracts projections onto existing basis vectors and
       normalises the residual.

Post Processing
~~~~~~~~~~~~~~~

``JenpyROQ.post_processing`` produces:

* pre-selection residual plots;
* empirical interpolation error plots;
* outlier-count plots;
* waveform comparison plots;
* representation-error plots;
* random validation tests;
* basis-parameter histograms;
* empirical-frequency histograms.
