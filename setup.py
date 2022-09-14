#!/usr/bin/env python
from setuptools import setup, find_packages

def find_version(path, varname="__version__"):
    """Parse the version metadata variable in the given file.
    """
    with open(path, 'r') as fobj:
        version_file = fobj.read()
    version_match = re.search(
        r"^{0} = ['\"]([^'\"]*)['\"]".format(varname),
        version_file,
        re.M,
    )
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")

setup(
    name='JenpyROQ',
    version=find_version(HERE / "JenpyROQ" / "__init__.py"),
    description='Construct reduced order quadratures bases and interpolants for gravitational wave data analysis.',
    author='Gregorio Carullo Sebastiano Bernuzzi Matteo Breschi Jacopo Tissino',
    author_email='gregorio.carullo@ligo.org',
    url = 'https://github.com/GCArullo/JenpyROQ',
    packages = find_packages(),
    requires = ['h5py', 'numpy', 'matplotlib']
)
