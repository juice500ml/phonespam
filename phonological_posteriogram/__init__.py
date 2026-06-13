"""Phone recognition and segmentation using phonological posteriograms."""

from .phone_model import PhoneModel
from .posteriogram import PhonologicalPosteriogram
from .recognizer import Recognizer, panphon_featmap
from .segmenter import Segmenter

try:
    from ._version import version as __version__
except ImportError:
    __version__ = "0.0.0"

__all__ = [
    "PhoneModel",
    "PhonologicalPosteriogram",
    "Recognizer",
    "Segmenter",
    "panphon_featmap",
    "__version__",
]
