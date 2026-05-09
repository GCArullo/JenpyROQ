# -*- coding: utf-8 -*-

import re
from pathlib import Path

from setuptools import find_packages, setup


HERE = Path(__file__).parent


def find_version(path, varname="__version__"):
    """Parse the version metadata variable in the given file."""
    with open(path, "r", encoding="utf-8") as fobj:
        version_file = fobj.read()
    version_match = re.search(
        r"^{0} = ['\"]([^'\"]*)['\"]".format(varname),
        version_file,
        re.M,
    )
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


def read_requirements(path):
    with open(path, "r", encoding="utf-8") as requires_file:
        return [
            line.strip()
            for line in requires_file
            if line.strip() and not line.lstrip().startswith("#")
        ]


with open(HERE / "pypi_description.rst", encoding="utf-8") as fobj:
    long_description = fobj.read()


setup(
    name="JenpyROQ",
    use_scm_version=True,
    description=(
        "Construct reduced order quadrature bases and interpolants "
        "for gravitational-wave data analysis."
    ),
    long_description=long_description,
    packages=find_packages(),
    extras_require={
        "docs": read_requirements(HERE / "docs" / "requirements.txt"),
        "mpi": ["mpi4py"],
    },
)
