import numpy as np
import pytest
from pathlib import Path

from lwa_daynight.settings import SettingsPair, FrequencyFactor, EQ_SPACING_HZ, BEAM_EQ_OFFSET

DATA = Path(__file__).resolve().parents[1] / "data"
DAY = DATA / "20251120a-settingsAll-day-FW7p5.mat"
NIGHT = DATA / "20251120f-settingsAll-night-FW7p5.mat"

S = 1.7782794100389228  # 10^(5/20)


@pytest.fixture(scope="module")
def pair():
    return SettingsPair(str(DAY), str(NIGHT))


def test_pair_shapes(pair):
    assert np.asarray(pair.day["coef"]).shape == (7, 4096)
    assert np.asarray(pair.night["coef"]).shape == (7, 4096)
    assert np.asarray(pair.day["settings"]).shape == (45, 16, 4)
    assert pair.group_counts.sum() == 704


def test_summary_consistency(pair):
    s = pair.summary()
    assert s["delay_identical"] is True
    assert s["groups_identical"] is True
    assert s["fft_shift_identical"] is True
    assert s["at2_identical"] is True
    assert s["at1_diff_unique_dB"] == [-5.0]


def test_ratio_uniform_below_break(pair):
    r = pair.group_ratios()
    # exactly 1.7783 (to ~2%) below the documented break at EQ channel 1186
    # (the transition itself spans the last few channels)
    assert np.allclose(r[:, :1180], S, rtol=2e-2)
    # structured above the break
    assert r[:, 1186:].std() > 0.02


def test_factor_anchors(pair):
    f = pair.factor()
    c = f.eval_power(np.array([20e6, 25e6, 27.99e6]))
    assert np.all(np.abs(c - S ** 2) < 0.01 * S ** 2)  # flat +5 dB region
    c50 = f.eval_power([50e6])[0]
    assert abs(10 * np.log10(c50) - 5.0) < 1.0  # within the structured band


def test_beam_offset_mapping(pair):
    f = pair.factor()
    c0 = f.eval_power([13398437.5])[0]  # beam channel 0
    assert np.isclose(c0, f.r_volt[BEAM_EQ_OFFSET] ** 2)
    # spacing sanity: eq channel of 13.3984375 MHz is exactly 560
    assert int(round(13398437.5 / EQ_SPACING_HZ)) == 560
    assert np.array_equal(f.eq_channel([13398437.5]), [560])


def test_inverse(pair):
    fd = pair.factor(to_state="day", from_state="night")
    fn = pair.factor(to_state="night", from_state="day")
    prod = fd.r_volt * fn.r_volt
    # uniform region reciprocates; structured region differs only by the
    # mean(x) vs mean(1/x) effect of averaging ratios over EQ groups (< 5%)
    assert np.allclose(prod[:1186], 1.0, rtol=2e-2)
    assert np.abs(prod[1186:] - 1).max() < 0.05


def test_range_raises(pair):
    f = pair.factor()
    with pytest.raises(ValueError):
        f.eval_power([110e6])  # above the 0-98 MHz EQ grid


def test_save_load_roundtrip(pair, tmp_path):
    f = pair.factor()
    p = tmp_path / "f.npz"
    f.save(str(p))
    g = FrequencyFactor.load(str(p))
    assert np.allclose(f.r_volt, g.r_volt)
    assert g.eq_offset == f.eq_offset
    assert g.provenance.get("weights") == "counts"
