#!/usr/bin/env python
from setuptools import setup, find_packages

setup(
    name='JenpyROQ',
    version='0.0.1',
    description='Construct reduced order quadratures bases and interpolants for gravitational wave data analysis.',
    author='Gregorio Carullo Sebastiano Bernuzzi Matteo Breschi Jacopo Tissino',
    author_email='gregorio.carullo@ligo.org',
    url = 'https://github.com/GCArullo/JenpyROQ',
    packages = find_packages(),
    requires = ['h5py', 'numpy', 'matplotlib']
)
