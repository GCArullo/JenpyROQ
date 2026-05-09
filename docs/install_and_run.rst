Install & Run
-------------

Install From Source
~~~~~~~~~~~~~~~~~~~

Starting from the repository root, install the package with:

.. code-block:: bash

   python -m pip install .

For editable development installs, use:

.. code-block:: bash

   python -m pip install -e .

Install documentation dependencies with:

.. code-block:: bash

   python -m pip install .[docs]

Install the MPI extra when using ``parallel = 2``:

.. code-block:: bash

   python -m pip install .[mpi]

Optional dependencies are needed only for the waveform families that use them.
See :doc:`waveform_models` before choosing an approximant.

Run
~~~

All production runs use an INI configuration file:

.. code-block:: bash

   JenpyROQ --config-file config.ini

The help message lists the configuration sections and option syntax:

.. code-block:: bash

   JenpyROQ --help

MPI Runs
~~~~~~~~

MPI execution is selected in the config file with:

.. code-block:: ini

   [Parallel]
   parallel    = 2
   n-processes = 128

Launch the job with the same task count:

.. code-block:: bash

   mpiexec -n 128 JenpyROQ --config-file config.ini

The code checks that ``n-processes`` is at least ``2`` when parallel execution
is active. For MPI, the master process uses the configured ``random-seed`` and
worker ranks add their rank to that seed. With ``random-seed = 170817``, rank
0 receives ``170817``, rank 1 receives ``170818``, and so on.
