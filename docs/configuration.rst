Configuration
-------------

``JenpyROQ`` reads INI files with six logical sections:

.. code-block:: ini

   [I/O]
   [Parallel]
   [Waveform_and_parametrisation]
   [ROQ]
   [Training_range]
   [Test_values]

The implementation defaults below are the values assigned in
``JenpyROQ/initialise.py`` before the config file is read.

I/O Section
~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 28 18 48

   * - Option
     - Default
     - Meaning
   * - ``output``
     - ``./``
     - Run directory where logs, copied config, ROQ arrays and plots are
       written.
   * - ``verbose``
     - ``1``
     - Enables stream logging when non-zero.
   * - ``debug``
     - ``0``
     - Activates additional basis-matrix inversion checks and creates
       ``Debug`` under the output directory.
   * - ``timing``
     - ``0``
     - Logs wall-clock timings for pre-selection, enrichment and validation
       steps.
   * - ``show-plots``
     - ``0``
     - Calls ``plt.show()`` after the run when enabled.
   * - ``post-processing-only``
     - ``0``
     - Skips basis construction, loads an existing ROQ from ``output`` and
       regenerates diagnostics.
   * - ``random-seed``
     - ``170817``
     - Initial seed for pseudo-random training-set generation.

Parallel Section
~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 28 18 48

   * - Option
     - Default
     - Meaning
   * - ``parallel``
     - ``0``
     - ``0`` uses the serial pool, ``1`` uses ``multiprocessing``, ``2`` uses
       MPI.
   * - ``n-processes``
     - ``4``
     - Number of worker processes requested by the selected pool.

When ``parallel`` is non-zero, ``n-processes`` must be at least ``2``. The code
does not infer this value from ``mpiexec``; keep the config and launch command
aligned.

Waveform & Parametrisation Section
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 20 46

   * - Option
     - Default
     - Meaning
   * - ``approximant``
     - ``teobresums-giotto``
     - Waveform family. Available values depend on installed optional
       libraries.
   * - ``spins``
     - ``aligned``
     - Selects active spin degrees of freedom: ``no-spins``, ``aligned`` or
       ``precessing``.
   * - ``tides``
     - ``0``
     - Adds ``lambda1`` and ``lambda2`` to the training domain.
   * - ``eccentricity``
     - ``0``
     - Adds ``ecc`` to the training domain.
   * - ``post-merger``
     - ``0``
     - Adds ``nrpmw-tcoll``, ``nrpmw-df2`` and ``nrpmw-phi``.
   * - ``dynamics``
     - ``0``
     - Adds ``TEOBResumS_a6c`` and ``TEOBResumS_cN3LO``.
   * - ``mc-q-par``
     - ``1``
     - Uses chirp mass and mass ratio as the sampled mass coordinates.
   * - ``m-q-par``
     - ``0``
     - Uses total mass and mass ratio as the sampled mass coordinates.
   * - ``spin-sph``
     - ``0``
     - Uses spherical spin coordinates. This is accepted only with
       ``spins = precessing``.
   * - ``f-min``
     - ``20.0``
     - Lower edge of the construction frequency grid in Hz.
   * - ``f-max``
     - ``2048.0``
     - Upper edge of the construction frequency grid in Hz.
   * - ``seglen``
     - ``128.0``
     - Segment length in seconds. The frequency spacing is ``1 / seglen``.

``mc-q-par`` and ``m-q-par`` are mutually exclusive. If both are set to
``0``, the code samples ``m1`` and ``m2`` directly.

Frequency Grid Size
~~~~~~~~~~~~~~~~~~~

The frequency-grid size is controlled by ``f-min``, ``f-max`` and ``seglen``.
The frequency spacing is:

.. container:: key-equation

   .. math::

      \Delta f = \frac{1}{\mathrm{seglen}}.

The code constructs the grid with:

.. code-block:: python

   np.arange(f_min, f_max + deltaF, deltaF)

If ``(f_max - f_min) / deltaF`` is an integer, the number of grid points is:

.. container:: key-equation

   .. math::

      N_f = (f_\mathrm{max} - f_\mathrm{min}) \times \mathrm{seglen} + 1.

For example, with ``f-min = 20``, ``f-max = 2048`` and ``seglen = 128``:

.. math::

   \Delta f = 1/128 = 0.0078125\,\mathrm{Hz},

.. math::

   (2048 - 20) / 0.0078125 = 2028 \times 128 = 259584,

so the grid has:

.. math::

   259584 + 1 = 259585

frequency samples. This count directly affects memory use, basis construction
time and the maximum possible ROQ speedup.

ROQ Section
~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 32 20 42

   * - Option
     - Default
     - Meaning
   * - ``gram-schmidt``
     - ``0``
     - When enabled, orthogonalises new basis elements against the known basis.
   * - ``basis-lin``
     - ``1``
     - Builds the linear waveform basis for ``h_+``.
   * - ``basis-qua``
     - ``1``
     - Builds the quadratic basis for ``|h_+|^2`` and uses it for quadratic
       diagnostics.
   * - ``pre-basis``
     - ``corners``
     - Starting basis strategy. Supported values are ``corners``,
       ``partial-pre-selected-basis``, ``pre-selected-basis`` and
       ``pre-enriched-basis``.
   * - ``tolerance-pre-basis-lin``
     - ``1e-8``
     - Projection-error threshold for the linear pre-selection loop.
   * - ``tolerance-pre-basis-qua``
     - ``1e-10``
     - Projection-error threshold for the quadratic pre-selection loop.
   * - ``n-pre-basis-lin``
     - ``80``
     - Maximum number of linear pre-selection elements, including the two
       initial ``corners`` seed elements.
   * - ``n-pre-basis-qua``
     - ``5``
     - Maximum number of quadratic pre-selection elements, including the two
       initial ``corners`` seed elements.
   * - ``n-pre-basis-search-iter``
     - ``80``
     - Number of random candidates evaluated when searching for a new
       pre-selection basis element.
   * - ``n-training-set-cycles``
     - ``4``
     - Number of enrichment cycles.
   * - ``training-set-sizes``
     - ``10000,100000,1000000,10000000``
     - Comma-separated training-set size for each enrichment cycle.
   * - ``training-set-n-outliers``
     - ``20,20,1,0``
     - Number of outliers allowed to remain at the end of each cycle.
   * - ``training-set-rel-tol``
     - ``0.1,0.1,0.05,0.3,1.0``
     - Relative factors multiplying ``tolerance-lin`` or ``tolerance-qua``.
   * - ``tolerance-lin``
     - ``1e-8``
     - Final interpolation-error scale for the linear basis.
   * - ``tolerance-qua``
     - ``1e-10``
     - Final interpolation-error scale for the quadratic basis.
   * - ``n-tests-post``
     - ``1000``
     - Number of random validation points after construction.
   * - ``minimum-speedup``
     - ``1.0``
     - Minimum allowed ratio ``len(full frequency grid) / len(ROQ nodes)``.

The three list options ``training-set-sizes``, ``training-set-n-outliers`` and
``training-set-rel-tol`` must have length equal to ``n-training-set-cycles``.
The default ``training-set-rel-tol`` currently contains five entries while the
default cycle count is four; production configs should set these lists
explicitly. For example:

.. code-block:: ini

   n-training-set-cycles   = 3
   training-set-sizes      = 10000,100000,100000
   training-set-n-outliers = 0,0,0
   training-set-rel-tol    = 0.1,1.0,1.0

Training Range Syntax
~~~~~~~~~~~~~~~~~~~~~

The training range section uses one ``-min`` and one ``-max`` key per active
parameter:

.. code-block:: ini

   [Training_range]
   mc-min      = 1.1968
   mc-max      = 1.1988
   q-min       = 1.0
   q-max       = 3.0
   lambda1-min = 5.0
   lambda1-max = 5000.0

The test-values section sets one representative point used in the waveform
comparison plots:

.. code-block:: ini

   [Test_values]
   mc      = 1.1975
   q       = 1.1
   lambda1 = 200

If a test value is outside its training range, the code logs a warning but does
not abort. If an upper bound is smaller than its lower bound, the run raises
``ValueError``.

Parameter Activation
~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 28 34 28

   * - Parameter names
     - Active when
     - Default range
   * - ``mc``, ``q``
     - ``mc-q-par = 1``
     - ``mc: [0.9, 1.4]``, ``q: [1.0, 3.0]``
   * - ``m``, ``q``
     - ``m-q-par = 1``
     - ``m: [2.0, 4.0]``, ``q: [1.0, 3.0]``
   * - ``m1``, ``m2``
     - ``mc-q-par = 0`` and ``m-q-par = 0``
     - ``m1: [1.0, 3.0]``, ``m2: [0.5, 2.0]``
   * - ``s1z``, ``s2z``
     - ``spins`` is ``aligned`` or ``precessing`` and ``spin-sph = 0``
     - ``[-0.5, 0.5]``
   * - ``s1x``, ``s1y``, ``s2x``, ``s2y``
     - ``spins = precessing`` and ``spin-sph = 0``
     - ``0.0`` to ``0.0``
   * - ``s1s1`` through ``s2s3``
     - ``spin-sph = 1``
     - magnitudes ``[0, 0.5]``, polar angles ``[0, pi]``, azimuths
       ``[0, 2*pi]``
   * - ``lambda1``, ``lambda2``
     - ``tides = 1``
     - ``[5.0, 5000.0]``
   * - ``ecc``
     - ``eccentricity = 1``
     - ``[0.0, 0.0]``
   * - ``TEOBResumS_a6c``, ``TEOBResumS_cN3LO``
     - ``dynamics = 1``
     - ``[-100.0, -20.0]``
   * - ``nrpmw-tcoll``, ``nrpmw-df2``, ``nrpmw-phi``
     - ``post-merger = 1``
     - ``[0, 3000]``, ``[-1e-5, 1e-5]``, ``[0, 2*pi]``

Mass Coordinates
~~~~~~~~~~~~~~~~

When ``mc-q-par = 1``, sampled ``mc`` and ``q`` are converted to component
masses by:

.. container:: key-equation

   .. math::

      m_2 = \mathcal{M} q^{-3/5} (1 + q)^{1/5}, \qquad m_1 = q m_2.

When ``m-q-par = 1``, sampled total mass ``m`` and mass ratio ``q`` are
converted by:

.. container:: key-equation

   .. math::

      m_1 = \frac{m q}{1 + q}, \qquad m_2 = \frac{m}{1 + q}.
