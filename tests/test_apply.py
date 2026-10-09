import numpy as np
import h5py
import pytest

from lwa_daynight.settings import FrequencyFactor, EQ_SPACING_HZ
from lwa_daynight.apply import correct_hdf5

NTIME = 12
NCHAN = 8
F0 = 13398437.5


def make_file(path, ntime=NTIME, nchan=NCHAN):
    with h5py.File(path, "w") as f:
        f.attrs["FileGenerator"] = "test"
        g = f.create_group("Observation1")
        g.attrs["Beam"] = 5
        t = np.zeros(ntime, dtype=[("int", "<i8"), ("frac", "<f8")])
        t["int"] = 1791471300 + np.arange(ntime)
        t["frac"] = 0.25
        g.create_dataset("time", data=t)
        tg = g.create_group("Tuning1")
        tg.create_dataset("freq", data=F0 + np.arange(nchan) * EQ_SPACING_HZ)
        for p in ("XX", "YY", "XY_real", "XY_imag"):
            tg.create_dataset(p, data=np.full((ntime, nchan), 2.0, dtype="f4"))
    return path


def test_apply_multiplies(tmp_path):
    inp = make_file(tmp_path / "in.h5")
    r = np.ones(4096)
    r[560:568] = 1.5
    fac = FrequencyFactor(r_volt=r)
    info = correct_hdf5(str(inp), str(tmp_path / "out.h5"), fac)
    assert info["rows"] == NTIME
    with h5py.File(tmp_path / "out.h5", "r") as f:
        xx = f["Observation1/Tuning1/XX"][...]
        yy = f["Observation1/Tuning1/YY"][...]
        xy = f["Observation1/Tuning1/XY_real"][...]
        assert np.allclose(xx, 2.0 * 1.5 ** 2)
        assert np.allclose(yy, xx)
        assert np.allclose(xy, xx)
        assert f.attrs["DAYNIGHT_CORR"] == "night->day"
        # time dataset preserved as compound
        assert f["Observation1/time"].dtype.names == ("int", "frac")


def test_apply_time_selection(tmp_path):
    inp = make_file(tmp_path / "in.h5")
    fac = FrequencyFactor(r_volt=np.ones(4096))
    t0 = 1791471303.0
    t1 = 1791471306.0
    info = correct_hdf5(str(inp), str(tmp_path / "out.h5"), fac, t0=t0, t1=t1)
    assert info["rows"] == 3
    with h5py.File(str(tmp_path / "out.h5"), "r") as f:
        assert f["Observation1/Tuning1/XX"].shape == (3, NCHAN)


def test_no_overwrite(tmp_path):
    inp = make_file(tmp_path / "in.h5")
    fac = FrequencyFactor(r_volt=np.ones(4096))
    correct_hdf5(str(inp), str(tmp_path / "out.h5"), fac)
    with pytest.raises(FileExistsError):
        correct_hdf5(str(inp), str(tmp_path / "out.h5"), fac)


def test_roundtrip_inverse(tmp_path):
    inp = make_file(tmp_path / "in.h5")
    r = np.linspace(0.9, 1.1, 4096)
    f1 = FrequencyFactor(r_volt=r)
    f2 = FrequencyFactor(r_volt=1.0 / r)
    correct_hdf5(str(inp), str(tmp_path / "mid.h5"), f1)
    correct_hdf5(str(tmp_path / "mid.h5"), str(tmp_path / "back.h5"), f2)
    with h5py.File(str(tmp_path / "back.h5"), "r") as f, h5py.File(str(inp), "r") as g:
        assert np.allclose(f["Observation1/Tuning1/XX"][...], g["Observation1/Tuning1/XX"][...], rtol=1e-6)
