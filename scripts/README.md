# scripts/

Demos and paper reproduction. **Nothing here is part of the `phonespam`
package**: it is not installed by `pip install phonespam`, not importable, and
the library never imports from it. The package depends on none of it; it all
depends on the package.

Everything here needs the training extra and a clone of this repo:

```bash
pip install -e ".[train]"
```

Plus, for anything touching the corpora: `TIMIT` (LDC-licensed, not
redistributable) and `VoxAngeles`. See [Data](../README.md#data) in the root
README for where to put them. A GPU is strongly recommended — feature
extraction and evaluation both run an S3M encoder over the full dataset.

## Reproducing the paper

| | |
| --- | --- |
| [`reproduce_main_result.sh`](reproduce_main_result.sh) | The whole pipeline: corpora → per-phone CSVs → S3M features → a trained `PhoneModel` → the reported metrics. The wavlm/24 run is the released [`juice500/wavlm-24-phonemodel`](https://huggingface.co/juice500/wavlm-24-phonemodel). Start here. |
| [`model_ablation.sh`](model_ablation.sh) | Trains across {wavlm, hubert, w2v2, xls-r} × layers {0, 3, …, 24}. |
| [`method_ablation.sh`](method_ablation.sh) | Structural ablations of the boundary-signal stack (single frame delta, stacked deltas, no mel-SVF). |
| [`efficiency_ablation.sh`](efficiency_ablation.sh) | Sample efficiency: refits on 1/2, 1/4, … 1/1024 of the TIMIT training utterances. |

Each stage skips when its output already exists, so the scripts are resumable;
`FORCE=1` redoes everything. The ablations reuse the feature pickle from
`model_ablation.sh` when it is present, so run that first to avoid re-encoding
the corpus several times. All four are configured by environment variables —
read the header comment of each for the full list.

```bash
TIMIT_ROOT=/path/to/TIMIT VOX_ROOT=/path/to/voxangeles \
  scripts/reproduce_main_result.sh
```

## Analysis

| | |
| --- | --- |
| [`evaluate_metrics.py`](evaluate_metrics.py) | Scores a trained artifact: boundary R-value, plus oracle and full-pipeline PER/TER/PFER on TIMIT test and VoxAngeles. `--model` also takes a Hub id. Set `CACHE_DIR` to memoize per-utterance inference across runs. |
| [`boundary_errors.py`](boundary_errors.py) | Where the segmenter deletes and inserts boundaries, attributed to phonetic context (silence edges, vowel–approximant transitions, and per-class split rates). |

## Notebooks

| | |
| --- | --- |
| [`example.ipynb`](example.ipynb) | One TIMIT utterance shown three ways — ground truth, recognition-only (frame-wise), and both heads — against the activation map. This is the qualitative figure in the paper. |
| [`visualizer.ipynb`](visualizer.ipynb) | SPAM heatmaps for individual utterances: pick feature channels, overlay predicted boundaries. |
| [`plots.ipynb`](plots.ipynb) | The layer-wise and sample-efficiency figures, from numbers pasted in by hand after running the ablations. Writes `plots/*.pdf`. |

The notebooks read sample audio from `plots/`, which is **not** in the repo —
`LDC93S1.wav`/`.phn` come from TIMIT and cannot be redistributed here. Copy
them out of your own TIMIT distribution (`TEST/DR1/FAKS0/SA1`, or the
`LDC93S1` sample) to run those cells.

They also start with `sys.path.append("..")`, which is only needed when working
from a clone without installing; `pip install -e .` makes it unnecessary.
