# spectquant/__init__.py

"""
SpectQuant: \
    A Specialized Package for Extracting Image Features \
    for Cardiac Amyloidosis Quantification on SPECT.
"""

__version__ = "0.1.0"
__author__ = "Markus Köfler"
__all__ = [
    "SUV",
    "TBR",
    "SegmentVolume",
    "Uptake",
    "SeptumVol",
    "create_segs"
]

from .create_segs import create_segs
from .suv import SUV
from .tbr import TBR
from .volume import SegmentVolume
from .uptake import Uptake
from .septum import SeptumVol
