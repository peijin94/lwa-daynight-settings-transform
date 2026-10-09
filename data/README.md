# Settings data provenance

Copies of the production OVRO-LWA settings files (firmware FW7p5 era),
taken from `lwacalim10:/home/pipeline/opsdata/` (world-readable on the LWA
operations cluster; these files are loaded with `mnc.settings.update()`, and
every load is appended to `arxAndF-settings.log` in the same directory).

| file | state | sha256 |
|---|---|---|
| `20251120a-settingsAll-day-FW7p5.mat` | day | `3212bdb2187c026b28eb3283dd087427234707b0819bd7d71b588d40a533e6d2` |
| `20251120f-settingsAll-night-FW7p5.mat` | night | `b9dd9fed8c93cb1e45e0610ec96ecc227e143088de3809619458b78dbc3d6834` |

MAT-file internal timestamps: day 1763761021.35 -> 2025-11-21 20:17:01 UTC;
night 1764088563.42 -> 2025-11-25 15:16:03 UTC.

Provenance comments inside the files:

* day: "20251120a: adjust 682 gains using 20251117 observations, AT2 only."
* night: "20251120f night PWC 7p5 +5dB: adjust 682 gains using 20251117
  observations, AT2 only." / "20251115f: PWC ignoring subbands starting at
  ch1186 28.352MHz. No freq ext. +5dB. Else same as 20251115b."

The older-era pair `20250612-settingsAll-{day,night}.mat` (10 dB regime) is
not bundled but was inspected during the derivation; see `docs/derivation.md`.
