
"""Setup for `spectquant` package."""

import os
import re
import subprocess
from typing import List

from setuptools import find_packages
from setuptools import setup
import platform

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


def _get_cuda_version():
    cuda_version = os.environ.get("CUDA_VERSION")
    if cuda_version is None:
        try:
            process = subprocess.run(
                ["nvidia-smi"], capture_output=True, text=True, check=True)
            output = process.stdout
            # extract cuda_version with regex
            match = re.search(r"CUDA Version: (\d+\.\d+)", output)
            if match:
                cuda_version = match.group(1)
            else:
                cuda_version = None
        except (subprocess.CalledProcessError, FileNotFoundError):
            cuda_version = None

    return cuda_version


def _get_cupy_dependency():
    cuda_version = _get_cuda_version()
    system = platform.system().lower()

    requirements_expr = cuda_version.split('.')[0] if cuda_version is not None else None
    # outputs e.g. 11x for cuda 11.0
    if requirements_expr is not None:
        requirements_expr += "x" if cuda_version else None
        return f'cupy-cuda{requirements_expr}'

    if cuda_version:
        if system == "linux":
            return f"cupy-cuda{cuda_version.replace('.', '')[:2]}"
        elif system == "darwin":  # macOS
            return f"cupy-cuda{cuda_version.replace('.', '')[:2]}-macosx_x86_64"
        elif system == "windows":
            return f"cupy-cuda{cuda_version.replace('.', '')[:2]}-windows_x86_64"
        else:
            # handle other platforms or fall back to CPU version
            return []
    else:
        return []  # fallback to CPU version


_VERSION = _get_version()
_README = _get_readme()
_INSTALL_REQUIREMENTS = _parse_requirements(os.path.join(
    _CURRENT_DIR, "requirements", "requirements.txt"))
_TEST_REQUIREMENTS = _parse_requirements(os.path.join(
    _CURRENT_DIR, "requirements", "requirements_test.txt"))
# _CUDA_VERSION = _get_cuda_version()
_GENERAL_CUPY_VERSION = _get_cupy_dependency()

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
    extras_require={
        # pip install spectquant[gpu]
        'gpu': _GENERAL_CUPY_VERSION, # leave empty list if no cupy version is found
    },
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
