"""Phone recognition and segmentation using phonological posteriograms."""

from .evaluation import SegmentationEvaluator, SegmentationUnit
from .phone_model import PhoneModel
from .posteriogram import PhonologicalPosteriogram
from .segmenter import Segmenter

try:
    from ._version import version as __version__
except ImportError:
    __version__ = "0.0.0"

__all__ = [
    "PhoneModel",
    "PhonologicalPosteriogram",
    "SegmentationEvaluator",
    "SegmentationUnit",
    "Segmenter",
    "__version__",
]
