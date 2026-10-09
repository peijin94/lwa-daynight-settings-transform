# Derivation: day/night settings transformation

Status: **validated against 2026-10 recordings** (see §4). Author: Hermes
agent session with P. Zhang, 2026-10-08/09.

## 1. What the settings pair contains

Production pair (since 2025-11, firmware `FW7p5`), on
`lwacalim10:/home/pipeline/opsdata/`:

* day: `20251120a-settingsAll-day-FW7p5.mat`
* night: `20251120f-settingsAll-night-FW7p5.mat`

Both are MATLAB v5 files loaded by `mnc.settings.update()` (see
`mnc_python/mnc/settings.py`). Contents (704 input signals = 11 SNAP2 boards
x 64; 45 ARX boards x 16 channels = 720 channel slots):

| key | shape | meaning |
|---|---|---|
| `coef` | (7, 4096) real | F-engine per-channel EQ voltage gains, one vector per cable/fiber group (`eq0`..`eq6` list their input membership, identical between the two files) |
| `settings` | (45, 16, 4) | ARX per-channel control: `(AT1 dB, AT2 dB, filter, FEE power)` |
| `delay_dsig` | (704,) | cable delays (identical day/night) |
| `fftShift` | scalar | 8191 both |

Measured differences between the two files:

* `AT1`: night = day **-5.000 dB**, uniformly on all 720 channels.
* `AT2`, filter (`HH`), FEE power: identical.
* `coef`: night = day / **1.778279...** (= 10^(5/20)); *exactly* uniform to
  ~0.4% below EQ channel 1186 (28.352 MHz), with a structured residual
  `R(nu)` above (per-group std 0.06-0.09 voltage ratio, extremes 0.84-1.19).
  The night file's own comment documents this: "PWC ignoring subbands
  starting at ch1186 28.352MHz. No freq ext. +5dB".
* The older-era pair (`20250612-*`) shows the same scheme with 10 dB:
  EQ ratio exactly 3.16228 (flat everywhere) and `AT1` night = day -10 dB.

## 2. What actually propagates to recorded beam power

The beam HDF5 `freq` axis starts at **13.3984375 MHz** with 23925.78125 Hz
spacing (196 MHz / 2 / 4096), i.e. beam channel `k` = EQ channel `560 + k`.
The EQ applies to all 3072 beam channels.

Empirical result from 2026-10-08 beam data:

* The **recorded power ratio** between a day-settings and a night-settings
  recording (same solar elevation, 2026-10-06/07 vs 2026-10-08, 15:20 UTC) is
  **+5.15 dB median** (29-86.5 MHz) and matches the squared `coef` ratio
  channel by channel: residual `r/C = +0.30 dB ± 0.72` (see §4).
* A mid-file ARX reload (2026-10-08 15:26 UTC, `AT1` re-loaded day values)
  produced **no step** in the recorded power beyond 0.1-0.4 dB solar drift.

Conclusion: the recorded beam power changes between states by
`C(f) = (coef_day/coef_night)^2`; the uniform `AT1` step does not propagate
to the recorded data. (Mechanism not established from these files alone —
`AT1` may sit outside the signal path or be compensated in the ARX/analog
chain; documented as an open question in the operations note.)

## 3. The correction factor

```
C(f) = (coef_day(f) / coef_night(f))^2          # power factor, night -> day
```

* Below 28.35 MHz: **flat +5.000 dB**.
* 30-85 MHz: 3.7-5.4 dB (the structured residual `R(nu)`), median ~+4.7 dB.
* Group combination: each EQ group's ratio weighted by input counts
  (variants differ by < 0.4 dB; use `--weights` to change).

Apply to power spectra (`XX`, `YY`, `XY_real`, `XY_imag` all scale by the
same real factor — `XY` is a voltage product, hence a power). The day->night
factor is the exact reciprocal.

## 4. Validation numbers (2026-10-06/07/08)

Residual `(day_ref / oct8_corrected)` after applying `C(f)`:

| band | Oct 7 ref | Oct 6 ref |
|---|---|---|
| 29-86.5 MHz | +0.30 dB, sigma 0.72 dB | +0.39 dB, sigma 0.77 dB |
| 14-28 MHz | -0.82 dB, sigma 2.03 dB (RFI/ionosphere-dominated) | +2.36 dB, sigma 2.64 dB |

Fine-scale ripple metric (std of log10 minus 101-channel running median,
29-86.5 MHz): raw Oct 8 15:20 = 0.0145 -> corrected 0.0104; day references
Oct 6/7 = 0.0067/0.0073; Oct 8 16:30 corrected = 0.0068 (equals the day
reference). Overall level check at 15:20: raw Oct 8 = Oct 7 - 5.31 dB;
corrected = Oct 7 - 0.47 dB.

Unrelated to settings: 2026-10-08 shows a large solar brightening
(+12 dB by 16:30 UTC, still present at 20:30), which dominates any
later-day comparisons.

## 5. Known limitations / open items

* `AT1` non-propagation verified only for the current-era pairs and this
  data set; re-verify when a new settings pair enters production.
* Low band (14-28 MHz) residuals are dominated by RFI and ionospheric
  day-to-day variability, not by the settings transform.
* The beam file `061321_2101000000005401018` (Oct 8, 21:00-21:54) has a
  corrupt `Observation1/time` dataset (recorder interrupted during the
  F-engine restart); it can be opened only with the SWMR read path and its
  time axis is unusable. Not repaired here (production data).
* Recording gap 2026-10-08 21:54-23:05 UTC (beam down during recovery), so
  the day-settings switch at 22:56 UTC has no direct data witness; the
  23:05-00:25 recovery recording is day-settings by construction.

## 6. Reproduction

```sh
# on a machine with the /nas7a beam data mounted (e.g. inti):
python scripts/extract_spectrum.py <beam.hdf5> out.npz 2026-10-08T15:20:00
python scripts/validate_oct8_2026.py     # residuals + ripple + diagnosis figure
python scripts/make_note_figures.py      # note figures
```
