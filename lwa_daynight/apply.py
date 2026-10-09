"""Apply a day/night correction factor to beam HDF5 power spectra.

Beam files (produced by the beam05 recorder) hold, under
``Observation1/Tuning1``: ``XX``, ``YY`` (auto-power) and ``XY_real`` /
``XY_imag`` (cross) spectra of shape (ntime, 3072), a ``freq`` axis in Hz,
plus a compound ``time`` dataset under ``Observation1``. All planes are
powers (voltage products), so all are corrected by the same real factor.
"""
from __future__ import annotations

import os

import h5py
import numpy as np

PLANES = ("XX", "YY", "XY_real", "XY_imag")


def open_hdf5_read(path: str) -> h5py.File:
    """Open read-only; fall back to SWMR read for files whose writer did not
    close cleanly (``file is already open for write`` error)."""
    try:
        return h5py.File(path, "r")
    except OSError as exc:
        if "already open for write" not in str(exc):
            raise
        return h5py.File(path, "r", swmr=True)


def read_time(f) -> tuple[np.ndarray, np.ndarray]:
    """Return (unix_seconds, valid_mask) for the beam time dataset."""
    t = f["Observation1/time"][:]
    if t.dtype.names:  # compound (int, frac) as written by lwahdf
        ti = t["int"].astype("f8")
        tv = ti + t["frac"].astype("f8")
        return tv, ti > 0
    tv = np.asarray(t, dtype="f8")
    return tv, np.isfinite(tv)


def correct_hdf5(infile: str, outfile: str, factor, planes=PLANES,
                 t0: float | None = None, t1: float | None = None,
                 chunk_rows: int = 2048, overwrite: bool = False) -> dict:
    """Write a corrected copy of ``infile`` with power spectra multiplied by
    ``factor.eval_power(freq)``. Returns a summary dict."""
    if os.path.exists(outfile) and not overwrite:
        raise FileExistsError(f"{outfile} exists (use overwrite=True)")

    with open_hdf5_read(infile) as fi:
        t, good = read_time(fi)
        lo = -np.inf if t0 is None else float(t0)
        hi = np.inf if t1 is None else float(t1)
        sel = np.where(good & (t >= lo) & (t < hi))[0]
        if sel.size == 0:
            raise ValueError("no time rows selected")
        freq = np.asarray(fi["Observation1/Tuning1/freq"][:], dtype=float)
        cpow = factor.eval_power(freq)

        with h5py.File(outfile, "w") as fo:
            for k, v in fi.attrs.items():
                fo.attrs[k] = v
            g = fo.create_group("Observation1")
            for k, v in fi["Observation1"].attrs.items():
                g.attrs[k] = v
            tg = g.create_group("Tuning1")
            for k, v in fi["Observation1/Tuning1"].attrs.items():
                tg.attrs[k] = v
            tg.create_dataset("freq", data=freq)
            g.create_dataset("time", data=fi["Observation1/time"][:][sel])
            for p in planes:
                dpath = f"Observation1/Tuning1/{p}"
                if dpath not in fi:
                    continue
                ds = fi[dpath]
                out = tg.create_dataset(p, shape=(sel.size, ds.shape[1]),
                                        dtype=ds.dtype, chunks=(min(chunk_rows, sel.size), ds.shape[1]))
                for i0 in range(0, sel.size, chunk_rows):
                    idx = sel[i0:i0 + chunk_rows]
                    j0, j1 = int(idx[0]), int(idx[-1]) + 1
                    blk = np.asarray(ds[j0:j1, :], dtype="f4")[idx - j0, :]
                    out[i0:i0 + idx.size, :] = blk * cpow[None, :].astype("f4")
            fo.attrs["DAYNIGHT_CORR"] = f"{factor.from_state}->{factor.to_state}"
            fo.attrs["DAYNIGHT_FACTOR_NOTE"] = factor.provenance.get("note", "")
    return {"input": infile, "output": outfile, "rows": int(sel.size),
            "t_first": float(t[sel[0]]), "t_last": float(t[sel[-1]]),
            "c_power_min": float(cpow.min()), "c_power_max": float(cpow.max())}
