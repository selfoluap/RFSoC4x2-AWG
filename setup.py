# Compatibility shim for PYNQ images that ship setuptools < 61.
# Those versions cannot read PEP 621 [project] metadata from pyproject.toml,
# causing pip to install a wheel named "UNKNOWN" with no packages inside.
# This setup.py provides explicit metadata so old setuptools can build a
# correct wheel.  With setuptools >= 61, pyproject.toml [project] is the
# primary source and this file is a harmless supplement.
from setuptools import find_packages, setup

setup(
    name="rfsoc4x2-awg",
    version="0.1.0",
    packages=find_packages(include=["firmware*"]),
    package_data={"firmware.notebooks": ["*.ipynb"]},
)
