# lwa-daynight-settings-transform

Correct OVRO-LWA **beamformed solar spectra recorded under night-time array
settings** so that they are equivalent to data recorded under day-time
settings.

## Why

The OVRO-LWA control system switches the array between two ARX + F-engine
settings files (a `*-settingsAll-day-*.mat` / `*-settingsAll-night-*.mat`
pair) at sunrise/sunset. Occasionally a recording is made while the "night"
settings are still loaded during daytime — e.g. on **2026-10-08 UTC**, when
the sunrise settings change failed and the day-time settings were only
restored at 22:56 UTC — leaving a day of solar observations recorded under
night-time settings.

The two settings files differ (for the current `FW7p5` pair) in:

| Item | Difference |
|---|---|
| F-engine EQ coefficients (`coef`, 7 groups x 4096 channels) | night = day / 1.7783, exactly, below EQ channel 1186 (28.38 MHz); structured residual above |
| ARX `AT1` attenuator | night = day - 5.000 dB (all 720 channels) |
| `AT2`, filter, FEE power, cable delays, FFT shift, EQ group membership | identical |

The recorded beam power changes between the two states by **exactly the
squared EQ coefficient ratio** — the uniform `AT1` step does *not* propagate
to recorded data (verified against beam recordings; see
`docs/derivation.md`). Hence the frequency-dependent correction factor

```
C(f) = [ coef_day(f) / coef_night(f) ]^2        (applied to power spectra)
```

## Quick start

```sh
python -m lwa_daynight info  --day data/20251120a-settingsAll-day-FW7p5.mat \
                             --night data/20251120f-settingsAll-night-FW7p5.mat

python -m lwa_daynight factor --day data/20251120a-settingsAll-day-FW7p5.mat \
                              --night data/20251120f-settingsAll-night-FW7p5.mat \
                              --from night --to day --out factor_night_to_day.npz

python -m lwa_daynight apply --factor factor_night_to_day.npz \
                             --input beam20261008/061321_14550000000032a503e \
                             --output 061321_14550000000032a503e.daycorr.hdf5
```

`apply` writes a corrected copy (or a selected time slice with
`--start/--stop`); the input file is never modified. All power planes
(`XX`, `YY`, `XY_real`, `XY_imag`) are multiplied by the same real factor.
Corrected files carry `night2day=True` (`False` for other directions) and
`DAYNIGHT_CORR='night->day'` in their root attributes.

## Validation (2026-10 data)

Same solar elevation (15:20 UTC), day-settings references from 2026-10-06/07
vs the night-settings recording of 2026-10-08:

* raw level offset: Oct 7/Oct 8 = **+5.15 dB** (band 29-86.5 MHz)
* after correction: residual **+0.30 dB** (Oct 7 ref.), **+0.39 dB** (Oct 6 ref.)
* fine-scale ripple (29-86.5 MHz, full channel resolution): raw **0.0145** ->
  corrected **0.0104**, vs day-reference 0.0067-0.0073 (log10 units);
  at 16:30 UTC the corrected ripple equals the day reference exactly (0.0068)

Figures: `docs/figs/` (factor, acceptance, sequence, full diagnosis).

## Data provenance

`data/*.mat` are copies of the production settings files from
`lwacalim10:/home/pipeline/opsdata/` (see `data/README.md` for hashes).

## Layout

```
lwa_daynight/       package (settings pair -> factor; HDF5 application; CLI)
scripts/            analysis scripts used for validation (spectra extraction, figures)
tests/              pytest suite (settings factor + synthetic HDF5 application)
docs/derivation.md  full derivation, incident timeline, open questions
data/               the day/night settings pair used in production since 2025-11
```

## License

MIT
