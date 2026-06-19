"""Phoible inventory helpers for recognizer vocab constraints.

The recognizer itself only accepts ``vocab=``. Use these helpers when a
Phoible inventory is the source of that vocab, then pass the returned tuple
directly to ``Recognizer.recognize(..., vocab=...)``.
"""

from __future__ import annotations

import functools
import os
import warnings

import pandas as pd

from .recognizer import _filter_panphon_known

_PHOIBLE_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "phoible.csv")


@functools.lru_cache(maxsize=1)
def _phoible():
    """Lazily load and cache the packaged Phoible inventory CSV."""
    df = pd.read_csv(
        _PHOIBLE_CSV,
        keep_default_na=False,
        na_values=[""],
        low_memory=False,
    )
    df["InventoryID"] = df["InventoryID"].astype(int)
    return df


def _inventory_dialect(series):
    """An inventory's SpecificDialect; never borrowed from another inventory."""
    specified = series.dropna()
    return specified.iloc[0] if not specified.empty else series.iloc[0]


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
        dialect_free = [i for i in inv_ids if pd.isna(dialect_per_id[i])]
        chosen = dialect_free[0] if dialect_free else inv_ids[0]
        id_dialect_lines = ", ".join(f"{i}: {dialect_per_id[i]!r}" for i in inv_ids)
        warnings.warn(
            f"Phoible has {len(inv_ids)} inventories for {lang!r} "
            f"({id_dialect_lines}); picking {chosen} "
            f"(dialect {dialect_per_id[chosen]!r}). Pass `phoible_id=...` "
            f"to override. See https://phoible.org/languages/{glottocode}",
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
        raise ValueError(f"Phoible InventoryID {phoible_id} not found in the packaged phoible.csv.")
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
