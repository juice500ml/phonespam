"""Phone recognition and segmentation using phonological posteriograms.

The quickest path is :meth:`PhoneModel.transcribe`, which runs the whole
pipeline on an audio file::

    from phonespam import PhoneModel

    model = PhoneModel.from_pretrained("juice500/wavlm-24-phonemodel")
    for seg in model.transcribe("utt.wav"):
        print(f"{seg.start:.2f}-{seg.end:.2f}  {seg.label}")

The individual stages -- :class:`PhonologicalPosteriogram`,
:class:`Segmenter`, :class:`Recognizer` -- stay available for anyone who
needs to drive them separately.
"""

from .phoible import inventory_id_for_language, vocab_for_inventory, vocab_for_language
from .phone_model import PhoneModel, Segment, Spam
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
    "Segment",
    "Segmenter",
    "Spam",
    "inventory_id_for_language",
    "panphon_featmap",
    "vocab_for_inventory",
    "vocab_for_language",
    "__version__",
]
