"""Phoible inventory helpers for recognizer vocab constraints.

The recognizer itself only accepts ``vocab=``. Use these helpers when a
Phoible inventory is the source of that vocab, then pass the returned tuple
directly to ``Recognizer.recognize(..., vocab=...)``.

The inventory table is not shipped inside the wheel -- it is a ~26 MB CSV. It
is downloaded from the PHOIBLE repository on first use and cached on disk
(see :func:`phoible_csv_path`). The download is pinned to one upstream commit
and checksum-verified, so a given phonespam release always resolves the same
inventories.

PHOIBLE is distributed under CC BY-SA 3.0 and is *not* covered by phonespam's
MIT license:

    Moran, Steven & McCloy, Daniel (eds.) 2019. PHOIBLE 2.0. Jena: Max Planck
    Institute for the Science of Human History. http://phoible.org
    DOI: 10.5281/zenodo.2626687
"""

from __future__ import annotations

import functools
import hashlib
import os
import tempfile
import warnings
from pathlib import Path

import pandas as pd
import requests

from .recognizer import _filter_panphon_known

# Pinned upstream snapshot: https://github.com/phoible/dev/tree/master/data
_PHOIBLE_COMMIT = "614c823cdf76127550fb2180bdf27a051c2972d5"
_PHOIBLE_URL = f"https://raw.githubusercontent.com/phoible/dev/{_PHOIBLE_COMMIT}/data/phoible.csv"
_PHOIBLE_SHA256 = "0816e698563b68ec6a309bab404a06dfb221d4334aa5ffcbc0c18f2bd01844b8"
_PHOIBLE_BYTES = 26455818


def _cache_root() -> Path:
    """Base directory for phonespam's downloaded data."""
    override = os.environ.get("PHONESPAM_CACHE_DIR")
    if override:
        return Path(override).expanduser()
    xdg = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg).expanduser() if xdg else Path.home() / ".cache"
    return base / "phonespam"


def _download_phoible(dest: Path) -> None:
    """Fetch the pinned phoible.csv to ``dest``, atomically and verified.

    Uses ``requests`` rather than ``urllib`` so the download goes through
    certifi's CA bundle; conda environments routinely ship an OpenSSL whose
    default CA path resolves to nothing, which breaks ``urllib`` alone.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    warnings.warn(
        f"Downloading the PHOIBLE inventory table (~{_PHOIBLE_BYTES // 1024 // 1024} MB) "
        f"to {dest}. This happens once; set PHONESPAM_CACHE_DIR to relocate the cache, "
        f"or PHONESPAM_PHOIBLE_CSV to point at a copy you already have.",
        stacklevel=3,
    )
    # Download to a unique temp file in the destination directory so a
    # concurrent process never observes a partial file, then rename.
    fd, tmp_name = tempfile.mkstemp(dir=dest.parent, prefix=".phoible-", suffix=".part")
    tmp = Path(tmp_name)
    try:
        digest = hashlib.sha256()
        with os.fdopen(fd, "wb") as out:
            with requests.get(_PHOIBLE_URL, stream=True, timeout=60) as resp:
                resp.raise_for_status()
                for chunk in resp.iter_content(chunk_size=1 << 20):
                    out.write(chunk)
                    digest.update(chunk)
        got = digest.hexdigest()
        if got != _PHOIBLE_SHA256:
            raise RuntimeError(
                f"Checksum mismatch for the downloaded PHOIBLE table: expected "
                f"{_PHOIBLE_SHA256}, got {got}. Retry; if it persists, the pinned URL "
                f"{_PHOIBLE_URL} may be serving different bytes."
            )
        os.replace(tmp, dest)
    except requests.RequestException as e:
        raise RuntimeError(
            f"Could not download the PHOIBLE inventory table from {_PHOIBLE_URL}: {e}. "
            f"Fetch it manually and point PHONESPAM_PHOIBLE_CSV at the file."
        ) from e
    finally:
        if tmp.exists():
            tmp.unlink()


def phoible_csv_path(*, download: bool = True) -> Path:
    """Path to the local phoible.csv, downloading it on first use.

    ``PHONESPAM_PHOIBLE_CSV`` overrides everything and is used verbatim, which
    is the escape hatch for offline machines and for pinning a different
    PHOIBLE snapshot. Otherwise the pinned snapshot is cached under
    ``PHONESPAM_CACHE_DIR`` (or ``$XDG_CACHE_HOME/phonespam``, or
    ``~/.cache/phonespam``), keyed by upstream commit so that bumping the pin
    never collides with an older copy.

    Pass ``download=False`` to get the cache path without fetching anything;
    the file may not exist.
    """
    override = os.environ.get("PHONESPAM_PHOIBLE_CSV")
    if override:
        path = Path(override).expanduser()
        if not path.exists():
            raise FileNotFoundError(
                f"PHONESPAM_PHOIBLE_CSV points at {path}, which does not exist."
            )
        return path

    dest = _cache_root() / "phoible" / _PHOIBLE_COMMIT / "phoible.csv"
    if download and not dest.exists():
        _download_phoible(dest)
    return dest


@functools.lru_cache(maxsize=1)
def _phoible():
    """Lazily load and cache the Phoible inventory CSV."""
    df = pd.read_csv(
        phoible_csv_path(),
        keep_default_na=False,
        na_values=[""],
        low_memory=False,
    )
    df["InventoryID"] = df["InventoryID"].astype(int)
    return df


def _inventory_dialect(series):
    """An inventory's SpecificDialect, or None if it has none.

    PHOIBLE writes a missing dialect as either an empty cell or the literal
    string ``"NA"``; both count as missing. Never borrowed from another
    inventory.
    """
    specified = series[series.notna() & (series != "NA")]
    return specified.iloc[0] if not specified.empty else None


@functools.cache
def inventory_id_for_language(lang: str) -> int:
    """Resolve a language name, ISO6393 code, or Glottocode to one InventoryID."""
    df = _phoible()
    target = str(lang).strip().lower()
    for col in ("LanguageName", "ISO6393", "Glottocode"):
        mask = df[col].astype(str).str.lower() == target
        rows = df[mask]
        if rows.empty:
            continue
        inv_ids = sorted(int(i) for i in rows["InventoryID"].unique())
        if len(inv_ids) == 1:
            return inv_ids[0]

        glottocode = rows["Glottocode"].iloc[0]
        dialect_per_id = {
            int(rid): _inventory_dialect(rsub["SpecificDialect"])
            for rid, rsub in rows.groupby("InventoryID")
        }
        dialect_free = [i for i in inv_ids if dialect_per_id[i] is None]
        chosen = dialect_free[0] if dialect_free else inv_ids[0]
        id_dialect_lines = ", ".join(f"{i}: {dialect_per_id[i]!r}" for i in inv_ids)
        warnings.warn(
            f"Phoible has {len(inv_ids)} inventories for {lang!r} "
            f"({id_dialect_lines}); picking {chosen} "
            f"(dialect {dialect_per_id[chosen]!r}). Use "
            f"`vocab_for_inventory(<InventoryID>)` to pick a different one. "
            f"See https://phoible.org/languages/{glottocode}",
            stacklevel=2,
        )
        return chosen
    raise ValueError(
        f"Language {lang!r} not found in Phoible (LanguageName / ISO6393 / Glottocode)."
    )


@functools.cache
def vocab_for_inventory(phoible_id: int, *, phoneme: bool = False) -> tuple[str, ...]:
    """Return one Phoible inventory as a panphon-filtered recognizer vocab."""
    df = _phoible()
    rows = df[df["InventoryID"] == int(phoible_id)]
    if rows.empty:
        raise ValueError(f"Phoible InventoryID {phoible_id} not found in phoible.csv.")
    if phoneme:
        phones = sorted(p for p in rows["Phoneme"].dropna().unique() if p != "NA")
    else:
        phones_set: set = set()
        for s in rows["Allophones"].dropna():
            phones_set.update(tok for tok in s.split() if tok != "NA")
        phones = sorted(phones_set)

    return _filter_panphon_known(tuple(phones), context=f"InventoryID {phoible_id}")


def vocab_for_language(lang: str, *, phoneme: bool = False) -> tuple[str, ...]:
    """Resolve a language and return its Phoible inventory as ``vocab``."""
    return vocab_for_inventory(inventory_id_for_language(lang), phoneme=phoneme)
