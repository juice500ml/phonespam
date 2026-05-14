"""Phone recognition and segmentation using phonological posteriograms."""

from .evaluation import SegmentationEvaluator, SegmentationUnit
from .model import PhonologicalVectors, Segmenter, SilenceHandler
from .pretrained import PhonologicalPosteriogram

try:
    from ._version import version as __version__
except ImportError:
    __version__ = "0.0.0"

__all__ = [
    "PhonologicalPosteriogram",
    "PhonologicalVectors",
    "SegmentationEvaluator",
    "SegmentationUnit",
    "Segmenter",
    "SilenceHandler",
    "__version__",
]
