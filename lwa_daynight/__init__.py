"""OVRO-LWA day/night settings transformation.

Derive a frequency-dependent correction factor from a day/night pair of
F-engine settings files (``*-settingsAll-*.mat``) and apply it to beamformed
power spectra so that night-settings recordings become equivalent to
day-settings recordings.

Empirical core (validated 2026-10, see docs/derivation.md): the recorded
beam power ratio between the two settings states equals the squared ratio of
the F-engine EQ coefficient vectors; the uniform ARX AT1 step in the files
does not propagate to recorded power.
"""

__version__ = "0.1.0"

from .settings import SettingsPair, FrequencyFactor, EQ_SPACING_HZ, BEAM_EQ_OFFSET  # noqa: F401
