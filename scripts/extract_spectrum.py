#!/usr/bin/env python3
"""Extract median beam spectra around given UTC times from a beam HDF5 file.

Usage:
  extract_spec.py FILE OUT.npz "2026-10-08T15:20:00" [...] [--win 120]

Saves npz with freq (Hz) and per-time median spectra for XX, YY, XY (present planes).
Prints strided (every 64th channel) medians for XX/YY.
"""
import sys
import datetime as dt
import numpy as np
import h5py

args = sys.argv[1:]
win = 120.0
if '--win' in args:
    i = args.index('--win')
    win = float(args[i + 1])
    del args[i:i + 2]
fn, outfn, times = args[0], args[1], args[2:]

t0s = [dt.datetime.fromisoformat(s).replace(tzinfo=dt.timezone.utc) for s in times]
unix0 = [t.timestamp() for t in t0s]


def read_time(f):
    t = f['Observation1/time'][:]
    if t.dtype.names:  # compound (int, frac)
        ti = t['int'].astype('float64')
        tf = t['frac'].astype('float64')
        tv = ti + tf
        good = ti > 0
    else:
        tv = np.asarray(t, dtype='float64')
        good = np.isfinite(tv)
    return tv, good


with h5py.File(fn, 'r') as f:
    t, good = read_time(f)
    print(f"time: n={t.size} valid={good.sum()} first={t[good][0]:.3f} last={t[good][-1]:.3f}")
    tv = t[good]
    d = np.diff(tv)
    if d.size:
        print(f"  cadence: med={np.median(d)*1000:.2f} ms")
    fr = np.asarray(f['Observation1/Tuning1/freq'][:], dtype=float)
    planes = [p for p in ('XX', 'YY', 'XY_real', 'XY_imag') if f'Observation1/Tuning1/{p}' in f]
    out = {'freq': fr}
    for u0, s in zip(unix0, times):
        sel = np.where(good & (t >= u0 - win) & (t <= u0 + win))[0]
        print(f"== {s}: n_sel={sel.size}")
        if sel.size == 0:
            continue
        for p in planes:
            lo, hi = int(sel[0]), int(sel[-1]) + 1
            block = f[f'Observation1/Tuning1/{p}'][lo:hi, :]  # contiguous read
            dd = np.asarray(block, dtype=float)[sel - lo, :]
            fin = np.isfinite(dd)
            ok = fin.mean(axis=1) > 0.5
            if ok.sum() == 0:
                continue
            m = np.nanmedian(dd[ok], axis=0)
            key = s.replace(':', '').replace('-', '')
            out[f'{p}_{key}'] = m
            if p in ('XX', 'YY'):
                nz = (m > 0) & np.isfinite(m)
                sstr = ','.join('%.3g' % v for v in m[::64])
                print(f"  {p}: rows_used={ok.sum()}/{sel.size} nonzero={nz.sum()}/{m.size} med={np.nanmedian(m[nz]):.4g}")
                print(f"    stride64: {sstr}")
np.savez(outfn, **out)
print("saved", outfn)
