Examples
--------

Runnable examples live under ``config_files``.

Run any example with:

.. code-block:: bash

   python -m JenpyROQ --config-file path/to/config.ini

Run a short implementation check with:

.. code-block:: bash

   JenpyROQ --config-file config_files/Test_configs/config_test_IMRPv2.ini

The test configurations under ``config_files/Test_configs`` are intentionally
small. They are designed to finish quickly and exercise the code path, not to
produce production-quality bases.

Test Configs
~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 38 52

   * - File
     - Description
   * - ``config_test_IMRPv2.ini``
     - LALSimulation ``IMRPhenomPv2`` aligned-spin binary black-hole test with
       multiprocessing and a narrow ``mc``/``q`` range.
   * - ``config_test_TEOB-SPA.ini``
     - ``teobresums-giotto`` aligned-spin test over a binary black-hole-like
       domain with tidal parameters enabled.
   * - ``config_test_TEOB-SPA_dynamics.ini``
     - ``teobresums-giotto`` BNS-like test with TEOB dynamics parameters
       enabled.
   * - ``config_test_MLGW-BNS.ini``
     - bajes ``mlgw-bns`` BNS inspiral test with tidal parameters.
   * - ``config_test_MLGW-BNS_standalone_test.ini``
     - Direct ``mlgw_bns`` standalone wrapper test over the same narrow BNS
       domain.
   * - ``config_test_MLGW-BNS-NRPMw.ini``
     - ``mlgw-bns-nrpmw`` hybrid inspiral/post-merger test with NRPMw
       post-merger parameters.
   * - ``config_test_MLGW-BNS-NRPMw-recal.ini``
     - Recalibrated ``mlgw-bns-nrpmw`` test. Extra recalibration parameters
       are included only when bajes exposes their names and bounds.
   * - ``config_test_NRPMw.ini``
     - Attached ``nrpmw`` test using total mass and mass ratio coordinates.
   * - ``config_test_NRPMw-recal.ini``
     - Recalibrated attached ``nrpmw`` test.
   * - ``config_test_NRPMw-merger.ini``
     - Merger-only ``nrpmw`` test with ``seglen = 0.080`` and
       ``f-min = 1024`` Hz.
   * - ``config_test_NRPMw-recal-merger.ini``
     - Recalibrated merger-only ``nrpmw`` test.
   * - ``config_test_TEOB-SPA-NRPMw.ini``
     - ``teobresums-spa-nrpmw`` inspiral/post-merger test.
   * - ``config_test_TEOB-SPA-NRPMw-recal.ini``
     - Recalibrated ``teobresums-spa-nrpmw`` test.

Production-Style Config
~~~~~~~~~~~~~~~~~~~~~~~

``config_files/config_MLGW-BNS_LVK_GW170817_release.ini`` is closer to a
production setup. It uses:

.. code-block:: ini

   output      = /home/gregorio.carullo/JenpyROQ/data/MLGW-BNS_LVK_GW170817
   parallel    = 2
   n-processes = 128
   approximant = mlgw-bns
   f-min       = 23
   f-max       = 2000
   seglen      = 128

The frequency spacing is:

.. math::

   \Delta f = 1 / 128 = 0.0078125\,\mathrm{Hz}.

The number of full-grid samples is:

.. math::

   (2000 - 23) / 0.0078125 + 1 = 1977 \times 128 + 1 = 253057.

It requests three enrichment cycles:

.. code-block:: ini

   training-set-sizes      = 10000,100000,100000
   training-set-n-outliers = 0,0,0
   training-set-rel-tol    = 0.1,1.0,1.0

and validation on ``500000`` random points. It requires the optional waveform
backend, MPI and substantial runtime.

Minimal Config Template
~~~~~~~~~~~~~~~~~~~~~~~

This template is suitable for a short smoke test after installing the required
waveform backend:

.. code-block:: ini

   [I/O]
   output      = smoke_test_roq
   verbose     = 1
   timing      = 1
   random-seed = 170817

   [Parallel]
   parallel    = 0
   n-processes = 4

   [Waveform_and_parametrisation]
   approximant = IMRPhenomPv2
   spins       = aligned
   f-min       = 50
   f-max       = 1024
   seglen      = 1

   [ROQ]
   basis-lin               = 1
   basis-qua               = 1
   n-pre-basis-search-iter = 50
   n-pre-basis-lin         = 4
   n-pre-basis-qua         = 2
   tolerance-pre-basis-lin = 1e-4
   tolerance-pre-basis-qua = 1e-4
   n-training-set-cycles   = 2
   training-set-sizes      = 1000,10000
   training-set-n-outliers = 2,0
   training-set-rel-tol    = 0.1,1.0
   tolerance-lin           = 1e-4
   tolerance-qua           = 1e-4
   n-tests-post            = 500

   [Training_range]
   mc-min     = 30
   mc-max     = 31
   q-min      = 1.0
   q-max      = 1.2
   s1z-min    = 0.0
   s1z-max    = 0.1
   s2z-min    = 0.0
   s2z-max    = 0.1
   phiref-min = 0.0
   phiref-max = 0.0
   iota-min   = 0.0
   iota-max   = 0.0

   [Test_values]
   mc     = 30.5
   q      = 1.1
   s1z    = 0.05
   s2z    = 0.05
   iota   = 0.0
   phiref = 0.0

Scaling Up
~~~~~~~~~~

* **Choose the waveform approximant** - install its optional backend first.
* **Set the frequency grid** - choose ``f-min``, ``f-max`` and ``seglen`` to
  match the intended likelihood grid.
* **Define the training box** - use the smallest domain that still covers the
  intended prior support.
* **Increase search depth first** - raise ``n-pre-basis-search-iter`` before
  increasing the number of enrichment cycles.
* **Tighten tolerances after profiling** - adjust ``tolerance-lin`` and
  ``tolerance-qua`` only after checking runtime and basis dimension.
* **Raise validation coverage** - set ``n-tests-post`` high enough that
  validation probes the full active domain.
* **Keep a speedup target** - keep ``minimum-speedup`` above the threshold
  required by the downstream inference use case.
