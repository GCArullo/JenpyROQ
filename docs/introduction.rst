Introduction
------------

``JenpyROQ`` builds frequency-domain reduced order quadrature data for fast
parameter estimation. The code samples a user-defined waveform parameter
space, constructs reduced bases for the linear and quadratic waveform terms,
selects empirical frequency nodes, and writes the arrays needed by downstream
inference codes.

Core References
~~~~~~~~~~~~~~~

The original algorithmic reference for the project is the ``PyROQ`` paper:

.. code-block:: bibtex

   @article{Qi:2020lfr,
     author = "Qi, Hong and Raymond, Vivien",
     title = "{Python-based reduced order quadrature building code for fast gravitational wave inference}",
     eprint = "2009.13812",
     archivePrefix = "arXiv",
     primaryClass = "gr-qc",
     doi = "10.1103/PhysRevD.104.063031",
     journal = "Phys. Rev. D",
     volume = "104",
     number = "6",
     pages = "063031",
     year = "2021"
   }

The paper introducing the ``JenpyROQ`` extensions is:

.. code-block:: bibtex

   @article{Tissino:2022thn,
     author = "Tissino, Jacopo and Carullo, Gregorio and Breschi, Matteo and Gamba, Rossella and Schmidt, Stefano and Bernuzzi, Sebastiano",
     title = "{Combining effective-one-body accuracy and reduced-order-quadrature speed for binary neutron star merger parameter estimation with machine learning}",
     eprint = "2210.15684",
     archivePrefix = "arXiv",
     primaryClass = "gr-qc",
     doi = "10.1103/PhysRevD.107.084037",
     journal = "Phys. Rev. D",
     volume = "107",
     number = "8",
     pages = "084037",
     year = "2023"
   }

We defer to this paper for details on the implemented algorithm and pointers
to the literature.

External libraries
~~~~~~~~~~~~~~~~~~

``JenpyROQ`` relies on a number of open-source packages and optional waveform
backends. If you use the software in your publications, please cite the
references found at these links where relevant:

.. raw:: html

   <div class="resource-chips">
     <a class="resource-chip" href="https://numpy.org/citing-numpy/"><span class="resource-mark resource-mark-cite">Cite</span><span>numpy</span></a>
     <a class="resource-chip" href="https://www.h5py.org/"><span class="resource-mark resource-mark-docs">Docs</span><span>h5py</span></a>
     <a class="resource-chip" href="https://github.com/matplotlib/matplotlib"><span class="resource-mark resource-mark-github">GitHub</span><span>matplotlib</span></a>
     <a class="resource-chip" href="https://seaborn.pydata.org/"><span class="resource-mark resource-mark-docs">Docs</span><span>seaborn</span></a>
     <a class="resource-chip" href="https://mpi4py.readthedocs.io/en/stable/citing.html"><span class="resource-mark resource-mark-cite">Cite</span><span>mpi4py</span></a>
     <a class="resource-chip" href="https://git.ligo.org/lscsoft/lalsuite"><span class="resource-mark resource-mark-gitlab">GitLab</span><span>lalsuite</span></a>
     <a class="resource-chip" href="https://bitbucket.org/eob_ihes/teobresums/src/master/README.md"><span class="resource-mark resource-mark-bitbucket">Bitbucket</span><span>TEOBResumS</span></a>
     <a class="resource-chip" href="https://pypi.org/project/mlgw-bns/"><span class="resource-mark resource-mark-pypi">PyPI</span><span>MLGW-BNS</span></a>
     <a class="resource-chip" href="https://github.com/matteobreschi/bajes"><span class="resource-mark resource-mark-github">GitHub</span><span>bajes</span></a>
   </div>

What The Code Produces
~~~~~~~~~~~~~~~~~~~~~~

A successful run writes:

.. rst-class:: jenpyroq-output-list

- The full frequency grid used during construction.
- Linear and/or quadratic pre-selection bases.
- Enriched bases and waveform parameters associated with basis elements.
- Empirical frequency nodes and integer node indices.
- Basis interpolants.
- An ``ROQ_metadata.txt`` file with the frequency-domain settings.
- Plots for basis parameters, empirical frequencies, interpolation errors,
  outlier counts, and waveform reconstruction checks.

The main output layout is described in :doc:`output_diagnostics`.
