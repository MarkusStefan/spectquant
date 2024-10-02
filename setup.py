
"""Setup for `spectquant` package."""

import os
from typing import List

from setuptools import find_packages
from setuptools import setup

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


def _get_readme() -> str:
    try:
        readme = open(
            os.path.join(_CURRENT_DIR, "README.md"), encoding="utf-8").read()
    except OSError:
        readme = ""
    return readme


def _get_version() -> None:
    with open(os.path.join(_CURRENT_DIR, "spectquant", "__init__.py")) as fp:
        for line in fp:
            if line.startswith("__version__") and "=" in line:
                version = line[line.find("=") + 1:].strip(" '\"\n")
                if version:
                    return version
        raise ValueError(
            "`__version__` not defined in `spectquant/__init__.py`")


def _parse_requirements(path) -> List[str]:

    with open(os.path.join(_CURRENT_DIR, path)) as f:
        return [
            line.rstrip()
            for line in f
            if not (line.isspace() or line.startswith("#"))
        ]


_VERSION = _get_version()
_README = _get_readme()
_INSTALL_REQUIREMENTS = _parse_requirements(os.path.join(
    _CURRENT_DIR, "requirements", "requirements.txt"))
_TEST_REQUIREMENTS = _parse_requirements(os.path.join(
    _CURRENT_DIR, "requirements", "requirements_test.txt"))

setup(
    name="spectquant",
    version=_VERSION,
    description="Specialized Package for Extracting Image Features for Cardiac Amyloidosis Quantification on SPECT.",
    long_description="\n".join(
        [_README]),
    long_description_content_type="text/markdown",
    author="MarkusStefan",
    author_email="markus.koefler11@gmail.com",
    license="MIT License",
    packages=find_packages(),
    install_requires=_INSTALL_REQUIREMENTS,
    tests_require=_TEST_REQUIREMENTS,
    url="https://github.com/MarkusStefan/spectquant",
    classifiers=[
            "Development Status :: 3 - Alpha",
            "Intended Audience :: Developers",
            "Intended Audience :: Science/Research",
            "License :: OSI Approved :: MIT License",
            "Operating System :: OS Independent",
            "Topic :: Scientific/Engineering",
            "Topic :: Scientific/Engineering :: Medical Science Apps.",
            "Topic :: Scientific/Engineering :: Image Processing",
            "Topic :: Scientific/Engineering :: Visualization",
            "Programming Language :: Python :: 3.7",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
            "Programming Language :: Python :: 3.11",
            "Programming Language :: Python :: 3.12"],
)
