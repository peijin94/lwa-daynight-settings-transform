"""Settings-pair loading and correction-factor derivation.

See docs/derivation.md for the full derivation and the 2026-10 validation.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import os
from dataclasses import dataclass, field

import numpy as np
from scipy.io import loadmat

#: F-engine EQ channel spacing: 196 MHz / 2 / 4096 (FW7p5 era).
EQ_SPACING_HZ = 196000000.0 / 8192.0  # 23925.78125 Hz

#: Number of EQ channels in the settings files.
EQ_CHANNELS = 4096

#: EQ channel index corresponding to beam-output channel 0 (FW7p5 era; the
#: beam HDF5 `freq` array starts at 13.3984375 MHz = 560 * EQ_SPACING_HZ).
BEAM_EQ_OFFSET = 560


def _sha256(path: str, blocksize: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(blocksize), b""):
            h.update(chunk)
    return h.hexdigest()


def load_settings_mat(path: str) -> dict:
    """Load one settings .mat file (scipy v5 format, squeeze_me)."""
    cfg = loadmat(path, squeeze_me=True)
    return {k: v for k, v in cfg.items() if not k.startswith("__")}


@dataclass
class FrequencyFactor:
    """Night->day (or day->night) frequency-dependent correction factor.

    ``r_volt`` is the voltage ratio (state_out / state_in) on the EQ channel
    grid; ``c_power = r_volt**2`` is the factor to apply to power spectra.
    """

    r_volt: np.ndarray                 # (n,) voltage ratio on EQ grid
    spacing_hz: float = EQ_SPACING_HZ
    eq_offset: int = BEAM_EQ_OFFSET
    from_state: str = "night"
    to_state: str = "day"
    provenance: dict = field(default_factory=dict)

    @property
    def c_power(self) -> np.ndarray:
        return self.r_volt ** 2

    def eq_channel(self, freq_hz) -> np.ndarray:
        """Nearest F-engine EQ channel index for frequencies given in Hz."""
        return np.round(np.asarray(freq_hz, dtype=float) / self.spacing_hz).astype(int)

    def eval_power(self, freq_hz, spacing_hz: float | None = None) -> np.ndarray:
        """Power factor on an arbitrary channel grid given frequencies in Hz.

        The factor lives on the F-engine EQ channel grid (channel k covers
        k * spacing_hz; 0 .. 4095 channels spanning 0-98 MHz). Each frequency
        is assigned the nearest EQ channel. Raises on frequencies beyond the
        grid rather than silently clipping.
        """
        sp = self.spacing_hz if spacing_hz is None else spacing_hz
        eq_ch = np.round(np.asarray(freq_hz, dtype=float) / sp).astype(int)
        bad = (eq_ch < 0) | (eq_ch >= self.r_volt.size)
        if bad.any():
            raise ValueError(
                f"{int(bad.sum())} frequencies map outside the EQ grid "
                f"[0, {(self.r_volt.size - 1) * sp / 1e6:.3f}] MHz"
            )
        return self.r_volt[eq_ch] ** 2

    def save(self, path: str) -> None:
        np.savez(
            path,
            r_volt=self.r_volt,
            spacing_hz=np.array(self.spacing_hz),
            eq_offset=np.array(self.eq_offset),
            from_state=np.array(self.from_state),
            to_state=np.array(self.to_state),
            **{f"meta_{k}": np.array(str(v)) for k, v in self.provenance.items()},
        )

    @classmethod
    def load(cls, path: str) -> "FrequencyFactor":
        z = np.load(path, allow_pickle=False)
        prov = {k[5:]: str(z[k]) for k in z.files if k.startswith("meta_")}
        return cls(
            r_volt=np.asarray(z["r_volt"], dtype=float),
            spacing_hz=float(z["spacing_hz"]),
            eq_offset=int(z["eq_offset"]),
            from_state=str(z["from_state"]),
            to_state=str(z["to_state"]),
            provenance=prov,
        )


class SettingsPair:
    """A day/night settings file pair (e.g. 20251120a-FW7p5 / 20251120f-FW7p5)."""

    def __init__(self, day_path: str, night_path: str):
        self.day_path = day_path
        self.night_path = night_path
        self.day = load_settings_mat(day_path)
        self.night = load_settings_mat(night_path)

    # ---------------------------------------------------------------- info
    @property
    def group_counts(self) -> np.ndarray:
        """Number of inputs in each EQ group (from the day file)."""
        return np.array([np.atleast_1d(self.day[f"eq{g}"]).size for g in range(7)], float)

    def summary(self) -> dict:
        s = {"day_file": os.path.basename(self.day_path),
             "night_file": os.path.basename(self.night_path)}
        cd, cn = np.asarray(self.day["coef"], float), np.asarray(self.night["coef"], float)
        ratio = cd / cn
        s["coef_ratio_median_below_eq1186"] = float(np.median(ratio[:, :1186]))
        s["coef_ratio_std_above_eq1186"] = float(ratio[:, 1186:].std())
        s["delay_identical"] = bool(np.array_equal(
            np.atleast_1d(self.day["delay_dsig"]), np.atleast_1d(self.night["delay_dsig"])))
        s["fft_shift_identical"] = self.day.get("fftShift") == self.night.get("fftShift")
        s["groups_identical"] = all(
            np.array_equal(np.atleast_1d(self.day[f"eq{g}"]), np.atleast_1d(self.night[f"eq{g}"]))
            for g in range(7))
        sd, sn = np.asarray(self.day["settings"], float), np.asarray(self.night["settings"], float)
        s["at1_diff_unique_dB"] = np.unique(np.round(sn[..., 0] - sd[..., 0], 3)).tolist()
        s["at2_identical"] = bool(np.array_equal(sd[..., 1], sn[..., 1]))
        return s

    # ------------------------------------------------------------- factors
    def group_ratios(self, to_state: str = "day", from_state: str = "night") -> np.ndarray:
        """(7, 4096) voltage ratio per EQ group, state_out / state_in from coef alone."""
        coef = {"day": np.asarray(self.day["coef"], float),
                "night": np.asarray(self.night["coef"], float)}
        num, den = coef[to_state], coef[from_state]
        return num / den

    def factor(self, to_state: str = "day", from_state: str = "night",
               weights: str = "counts") -> FrequencyFactor:
        """Correction factor between the two states.

        Weights for the per-group ratios combined into the beam-level factor:
        ``'counts'`` (default): weight each EQ group by its input count;
        ``'mean'``: unweighted mean over groups.
        """
        ratios = self.group_ratios(to_state, from_state)
        if weights == "counts":
            w = self.group_counts
            w = w / w.sum()
            r = (ratios * w[:, None]).sum(axis=0)
        elif weights == "mean":
            r = ratios.mean(axis=0)
        else:
            raise ValueError("weights must be 'counts' or 'mean'")
        prov = {
            "day_file": self.day_path,
            "night_file": self.night_path,
            "day_sha256": _sha256(self.day_path),
            "night_sha256": _sha256(self.night_path),
            "day_file_time": str(_dt.datetime.fromtimestamp(float(self.day["time"]), _dt.timezone.utc)),
            "night_file_time": str(_dt.datetime.fromtimestamp(float(self.night["time"]), _dt.timezone.utc)),
            "weights": weights,
            "note": ("power correction = (coef_out/coef_in)^2 combined over EQ groups; "
                     "ARX AT1 step does not propagate to recorded beam power (empirical, 2026-10)"),
        }
        return FrequencyFactor(r_volt=r, from_state=from_state, to_state=to_state, provenance=prov)
