Development Guide
-----------------

This page summarises the extension points that are most likely to matter when
adding a parameter, adding a waveform wrapper or changing the ROQ construction
logic.

Repository Map
~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 32 58

   * - Path
     - Role
   * - ``JenpyROQ/__main__.py``
     - Command-line entry point, output directory creation, logging, pool
       initialisation and top-level run loop.
   * - ``JenpyROQ/__init__.py``
     - Lightweight package metadata, including ``__version__`` for docs and
       runtime display.
   * - ``JenpyROQ/initialise.py``
     - Help text, config parsing, default values, active-parameter selection
       and training/test-value parsing.
   * - ``JenpyROQ/jenpyroq.py``
     - ROQ class, parameter transforms, basis construction, empirical-node
       selection and enrichment cycles.
   * - ``JenpyROQ/waveform_wrappers.py``
     - Waveform wrapper registry and model-specific input checks.
   * - ``JenpyROQ/linear_algebra.py``
     - Scalar product, normalisation, projection and Gram-Schmidt helper.
   * - ``JenpyROQ/post_processing.py``
     - Validation tests and diagnostic plots.
   * - ``JenpyROQ/parallel.py``
     - Serial, multiprocessing and MPI pool adapters.
   * - ``config_files/Test_configs``
     - Short implementation-check configs.
   * - ``setup.py`` and ``pyproject.toml``
     - Packaging metadata, dynamic version/readme configuration, core
       dependencies, optional extras and the installed ``JenpyROQ`` console
       script.

Adding A New Parameter
~~~~~~~~~~~~~~~~~~~~~~

Parameter activation is controlled in ``JenpyROQ/initialise.py``. A new
parameter needs four pieces of work.

.. rst-class:: jenpyroq-compact-steps

1. Add a default range and test value to ``default_params``.

   Example pattern:

   .. code-block:: python

      default_params["my_parameter"] = {
          "range": [0.0, 1.0],
          "test-value": 0.5,
      }

2. Update ``check_skip_parameter`` so the parameter is active only for the
   intended flags or approximants.

   If the parameter should be active only when ``my-flag = 1``, the check
   should return ``1`` when the flag is absent or false:

   .. code-block:: python

      if key == "my_parameter" and not input_par["Waveform_and_parametrisation"]["my-flag"]:
          return 1

3. Add any new config flag to the default
   ``input_par["Waveform_and_parametrisation"]`` dictionary and to the help
   string at the top of ``initialise.py``.

4. Use the parameter in the relevant waveform wrapper or parameter transform.
   The ROQ class passes active sampled parameters as a dictionary ``p`` into
   ``generate_waveform``.

Adding A Waveform Wrapper
~~~~~~~~~~~~~~~~~~~~~~~~~

The wrapper registry is the dictionary ``WfWrapper`` in
``JenpyROQ/waveform_wrappers.py``. A wrapper should:

* accept ``approximant`` and optional ``additional_waveform_params`` in
  ``__init__``;
* implement ``generate_waveform(self, p, deltaF, f_min, f_max, distance)``;
* return two arrays ``hp`` and ``hc`` on the grid
  ``np.arange(f_min, f_max + deltaF, deltaF)`` or an exactly compatible grid;
* validate unsupported parts of the requested parameter domain with explicit
  exceptions;
* register itself by assigning ``WfWrapper[approximant_name] = WrapperClass``.

For optional dependencies, follow the existing pattern:

.. code-block:: python

   try:
       import my_waveform_package

       class WfMyModel:
           ...

       WfWrapper["my-model"] = WfMyModel
   except ModuleNotFoundError:
       print("\nWarning: `my_waveform_package` module not found.\n")

ROQ Construction Changes
~~~~~~~~~~~~~~~~~~~~~~~~

The main algorithmic stages are:

* ``construct_corner_basis``;
* ``construct_preselection_basis``;
* ``interpolant_and_empirical_nodes``;
* ``roqs``;
* ``test_roq_error`` in ``post_processing.py``.

When changing one of these stages, update the stored-array format only if the
reader in ``post-processing-only`` mode is updated at the same time. Existing
resumability depends on the names and shapes of files under
``ROQ_data/linear`` and ``ROQ_data/quadratic``.

Testing Checklist
~~~~~~~~~~~~~~~~~

For a narrow code change, run at least one short config that exercises the
changed path. Examples:

.. code-block:: bash

   python -m JenpyROQ --config-file config_files/Test_configs/config_test_IMRPv2.ini
   python -m JenpyROQ --config-file config_files/Test_configs/config_test_MLGW-BNS.ini

For config-parser or docs changes, also run:

.. code-block:: bash

   python -m JenpyROQ --help
   sphinx-build -b html docs docs/_build/html

For MPI or multiprocessing changes, use a deliberately small training set
first, then repeat with a representative production-sized training set.
