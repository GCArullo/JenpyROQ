========
JenpyROQ
========

``JenpyROQ`` constructs reduced order quadrature bases and empirical
interpolation nodes for gravitational-wave waveform models. It extends the
original ``PyROQ`` workflow with configuration-file driven runs, modular
waveform wrappers, configurable pre-selection and enrichment cycles, and
diagnostic output for offline ROQ construction.

Install from a source checkout with:

.. code-block:: bash

   python -m pip install .

After installation, run with either:

.. code-block:: bash

   JenpyROQ --config-file config.ini

or:

.. code-block:: bash

   python -m JenpyROQ --config-file config.ini
