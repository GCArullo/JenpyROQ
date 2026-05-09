ROQ Workflow
------------

The main implementation lives in ``JenpyROQ/jenpyroq.py``. A run constructs
one or two bases depending on ``basis-lin`` and ``basis-qua``. The linear run
uses the plus polarisation ``h_+``. The quadratic run uses ``|h_+|^2`` in the
main construction and validates both plus and cross polarisations.

High-Level Flow
~~~~~~~~~~~~~~~

.. code-block:: text

   read configuration and create output directories
                  |
                  v
   initialise serial, multiprocessing or MPI pool
                  |
                  v
   seed pseudo-random generators
                  |
                  v
   build full frequency grid from f-min, f-max and seglen
                  |
                  v
   map active training parameters to vector indices
                  |
                  v
   construct or load a pre-selection basis
                  |
                  v
   run one or more enrichment cycles over random training sets
                  |
                  v
   compute empirical nodes and basis interpolant
                  |
                  v
   write ROQ arrays, metadata and diagnostic plots
                  |
                  v
   validate interpolation error on random test points

Inner Product And Normalisation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The scalar product used internally is:

.. container:: key-equation

   .. math::

      \langle a, b \rangle = 4 \Delta f \, \mathrm{Re}
      \sum_i a_i^* b_i.

With optional weights ``w_i``, both vectors are divided by
``sqrt(weights)`` before the sum. A vector is normalised as:

.. container:: key-equation

   .. math::

      \hat{h} = \frac{h}{\sqrt{\langle h, h \rangle}}.

If the norm is exactly zero, the code returns the original vector.

Pre-Selection Basis
~~~~~~~~~~~~~~~~~~~

The pre-selection stage is controlled by ``pre-basis`` and the
``n-pre-basis-*`` options:

* ``pre-basis = corners`` always starts from exactly two initial basis
  elements: one waveform evaluated with every active training parameter at its
  lower bound, and one waveform evaluated with every active training parameter
  at its upper bound.
* The corner initialisation therefore uses two opposite parameter-space
  points, not all ``2^N`` corners of the ``N``-dimensional training box.
* ``partial-pre-selected-basis`` loads the saved pre-selection basis and
  continues the pre-selection loop.
* ``pre-selected-basis`` loads the saved pre-selection basis and proceeds
  directly to enrichment.
* ``pre-enriched-basis`` loads the saved enriched basis and proceeds directly
  to the empirical-node and enrichment workflow.

Each pre-selection iteration then follows the same greedy search:

* Draw ``n-pre-basis-search-iter`` random points uniformly in the active
  training box.
* Generate one waveform per candidate point.
* Project each candidate onto the known basis.
* Compute the residual modulus for each projected candidate.
* Select the candidate with the largest residual modulus.
* Append that waveform as the next basis element.
* If ``gram-schmidt = 1``, append the normalised Gram-Schmidt residual instead
  of the raw normalised waveform.
* Store the current basis, waveform parameters and residual history under
  ``ROQ_data/linear`` or ``ROQ_data/quadratic``.
* Stop when the residual is below ``tolerance-pre-basis-lin`` or
  ``tolerance-pre-basis-qua``.
* Stop when the selected maximum basis size, ``n-pre-basis-lin`` or
  ``n-pre-basis-qua``, has been reached.

Empirical Interpolation Nodes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

After the pre-selection basis is available, the code computes empirical
interpolation nodes. The first node is the index where the absolute value of
the first basis vector is largest. For later basis vectors, the code solves a
small interpolation system, evaluates the residual between the current basis
vector and its interpolant, and selects the frequency index with the largest
absolute residual.

Repeated empirical nodes are not allowed. If a candidate residual maximum is
already present in the node list, the implementation logs a warning, manually
zeros the residual at previous nodes, and chooses the next maximum. At the end
of the node update, a repeated-node check raises ``ValueError`` if duplicates
remain.

The final interpolant is:

.. container:: key-equation

   .. math::

      B = E^T V^+,

where ``E`` is the basis matrix restricted to the number of empirical nodes and
``V^+`` is the pseudo-inverse of that basis evaluated at the empirical nodes.

Enrichment Cycles
~~~~~~~~~~~~~~~~~

Each enrichment cycle uses three aligned list entries:

.. code-block:: ini

   training-set-sizes      = 10000,100000,100000
   training-set-n-outliers = 0,0,0
   training-set-rel-tol    = 0.1,1.0,1.0

For cycle ``i`` the effective tolerance is:

.. container:: key-equation

   .. math::

      \epsilon_i = r_i \epsilon,

where ``r_i`` is ``training-set-rel-tol[i]`` and ``epsilon`` is
``tolerance-lin`` for the linear run or ``tolerance-qua`` for the quadratic
run.

For example, if ``tolerance-lin = 1e-4`` and the first relative tolerance is
``0.1``:

.. math::

   \epsilon_0 = 0.1 \times 10^{-4} = 10^{-5}.

The cycle then:

* draws ``training-set-sizes[i]`` random parameter points;
* computes the interpolation error for all current outliers;
* keeps only points with error above the effective tolerance;
* appends the worst represented point as a new basis element if too many
  outliers remain;
* repeats until ``len(outliers) <= training-set-n-outliers[i]``.

Minimum Speedup Guard
~~~~~~~~~~~~~~~~~~~~~

During enrichment the code checks:

.. container:: key-equation

   .. math::

      \frac{N_\mathrm{full}}{N_\mathrm{basis}} \geq
      \mathrm{minimum\mbox{-}speedup}.

If the ratio falls below ``minimum-speedup``, construction aborts. With
``N_full = 259585`` and ``minimum-speedup = 10``, the largest basis dimension
allowed by this guard is:

.. math::

   \left\lfloor 259585 / 10 \right\rfloor = 25958.

Validation
~~~~~~~~~~

After storing the ROQ arrays, ``test_roq_error`` draws ``n-tests-post`` random
points and evaluates the representation errors for plus and cross
polarisations. The plotted quantity is:

.. container:: key-equation

   .. math::

      2 \left(1 - \langle h, h_\mathrm{ROQ} \rangle \right).

The validation plot is written to:

.. code-block:: text

   Plots/Interpolation_errors_random_test_points_<lin|qua>.pdf
