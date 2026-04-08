"""Lightweight I/O helpers used across tools."""

from __future__ import annotations

from pathlib import Path


def read_csv_smart(file_path: str | Path):
    """Read a CSV with best-effort delimiter/encoding detection.

    - Tries fast comma-separated parsing first.
    - If it looks like a single-column mis-parse (e.g. header contains `;`),
      sniffs delimiter and retries.
    - Falls back to python-engine auto-sniffing when needed.
    """
    import csv

    import pandas as pd

    path = Path(file_path)

    def _read_sample() -> str:
        # Read a small chunk for delimiter sniffing. Use replacement to avoid decode errors.
        with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as f:
            return f.read(32_768)

    def _sniff_delimiter(sample: str) -> str | None:
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=[",", ";", "\t", "|"])
            return dialect.delimiter
        except csv.Error:
            return None

    # 1) Fast path: standard comma CSV.
    try:
        df = pd.read_csv(path)
        if df.shape[1] != 1:
            return df

        # If it's 1-column but the header contains typical delimiters, retry.
        col0 = str(df.columns[0])
        if any(sep in col0 for sep in (";", "\t", "|")):
            raise ValueError("suspected_delimiter_mismatch")
        return df
    except Exception:
        pass

    # 2) Sniff delimiter from sample then retry with common encodings.
    sample = _read_sample()
    delimiter = _sniff_delimiter(sample)

    # Prefer using a concrete delimiter with the fast engine.
    if delimiter:
        for enc in ("utf-8", "utf-8-sig", "latin-1"):
            try:
                return pd.read_csv(path, sep=delimiter, encoding=enc)
            except Exception:
                continue

    # 3) Last resort: let pandas/python engine try auto detection.
    last_exc: Exception | None = None
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(path, sep=None, engine="python", encoding=enc)
        except Exception as e:
            last_exc = e

    if last_exc:
        raise last_exc
    raise RuntimeError("Failed to read CSV for unknown reasons.")

