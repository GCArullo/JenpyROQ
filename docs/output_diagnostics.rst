Outputs & Diagnostics
---------------------

Each run writes enough metadata to inspect the construction afterwards. The
most important files are under ``ROQ_data`` and ``Plots``.

Run Directory
~~~~~~~~~~~~~

Before the ROQ object is initialised, the command creates the base output
directory. Its top-level layout is:

.. code-block:: text

   <output>/
     JenpyROQ.log
     git_info.txt
     <copied config file>
     ROQ_data/
       ROQ_metadata.txt
       full_frequencies.npy
       linear/
       quadratic/
     Plots/
       Basis_parameters/
       Waveform_comparisons/

The config file is copied into the run directory so the stored arrays can be
traced back to the exact input settings.

``JenpyROQ.log`` records parsed configuration values, default markers, basis
construction progress, outlier counts, validation settings and timing
information when ``timing = 1``.

``git_info.txt`` records the active git branch, latest commit hash, configured
git author and current diff. This file is written at config-read time.

ROQ Arrays
~~~~~~~~~~

The ``linear`` and ``quadratic`` subdirectories use the same file naming
pattern, with ``linear`` or ``quadratic`` inserted in the filename.

.. list-table::
   :header-rows: 1
   :widths: 38 52

   * - File
     - Meaning
   * - ``preselection_<type>_basis.npy``
     - Basis vectors produced by the pre-selection loop.
   * - ``preselection_<type>_basis_waveform_params.npy``
     - Parameter points associated with pre-selection basis vectors.
   * - ``preselection_<type>_basis_residual_modula.npy``
     - Largest projection residual found at each pre-selection step.
   * - ``basis_<type>.npy``
     - Enriched basis stored during the ROQ construction loop.
   * - ``basis_waveform_params_<type>.npy``
     - Parameter points associated with enriched basis vectors.
   * - ``basis_interpolant_<type>.npy``
     - Matrix used to reconstruct full-grid waveforms from empirical-node
       samples.
   * - ``empirical_frequencies_<type>.npy``
     - Physical frequencies selected as empirical interpolation nodes.
   * - ``empirical_nodes_<type>.npy``
     - Integer indices of those frequencies on ``full_frequencies.npy``.

The top-level ``ROQ_metadata.txt`` stores:

.. code-block:: text

   f-min 	 f-max 	 seglen
   <value> 	 <value> 	 <value>

The top-level ``full_frequencies.npy`` stores the original construction grid.

Reduction Factor
~~~~~~~~~~~~~~~~

The log reports:

.. code-block:: text

   <type> basis reduction factor:
   (Original freqs [N_full]) / (New freqs [N_roq]) = N_full / N_roq

This is a grid-size reduction diagnostic. The realised parameter-estimation
speedup also depends on likelihood implementation, waveform-call cost,
interpolation overhead and I/O.

Diagnostic Plots
~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 42 48

   * - Plot
     - Interpretation
   * - ``Plots/Preselection_residual_modulus_<lin|qua>.pdf``
     - Largest projection residual retained during pre-selection.
   * - ``Plots/Empirical_interpolation_error_<lin|qua>.pdf``
     - Maximum interpolation error recorded while adding enriched basis
       elements.
   * - ``Plots/Number_of_outliers_<lin|qua>.pdf``
     - Number of training points above the current cycle tolerance after each
       enrichment step.
   * - ``Plots/Frequencies_<lin|qua>.pdf``
     - Histogram of empirical frequencies.
   * - ``Plots/Interpolation_errors_random_test_points_<lin|qua>.pdf``
     - Post-construction validation errors on random test points.
   * - ``Plots/Representation_error_hp_<lin|qua>.pdf``
     - Fractional reconstruction error for ``h_+`` at the configured
       ``[Test_values]`` point.
   * - ``Plots/Representation_error_hc_<lin|qua>.pdf``
     - Fractional reconstruction error for ``h_x`` at the configured
       ``[Test_values]`` point.
   * - ``Plots/Waveform_comparisons/Waveform_comparison_*_<lin|qua>.pdf``
     - Full-grid waveform, ROQ reconstruction and empirical-node samples.
   * - ``Plots/Basis_parameters/Basis_parameters_<lin|qua>_<parameter>.pdf``
     - Distribution of parameter values selected for basis elements.

Post-Processing Only Mode
~~~~~~~~~~~~~~~~~~~~~~~~~

Set:

.. code-block:: ini

   [I/O]
   output = existing_run_directory
   post-processing-only = 1

In this mode the code skips construction and loads:

.. code-block:: text

   ROQ_data/<type>/empirical_frequencies_<type>.npy
   ROQ_data/<type>/empirical_nodes_<type>.npy
   ROQ_data/<type>/basis_interpolant_<type>.npy
   ROQ_data/<type>/basis_waveform_params_<type>.npy

If an old run lacks ``empirical_nodes_<type>.npy``, the code recovers node
indices by searching ``full_frequencies.npy`` for the stored empirical
frequencies.

Practical Checks
~~~~~~~~~~~~~~~~

Before using a basis in inference, inspect:

* validation errors relative to ``tolerance-lin`` and ``tolerance-qua``;
* whether outlier counts reached the requested values in every enrichment
  cycle;
* empirical-frequency histograms for unexpected clustering at boundaries;
* waveform-comparison plots for the configured ``[Test_values]`` point;
* basis-parameter histograms for gaps in the intended training domain.
