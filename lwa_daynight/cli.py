"""Command-line interface: ``python -m lwa_daynight <cmd>``."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys


def _iso(s: str) -> float:
    t = dt.datetime.fromisoformat(s)
    if t.tzinfo is None:
        t = t.replace(tzinfo=dt.timezone.utc)
    return t.timestamp()


def main(argv=None):
    ap = argparse.ArgumentParser(prog="lwa-daynight",
                                 description="OVRO-LWA day/night settings transform")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("info", help="summarize a day/night settings pair")
    p.add_argument("--day", required=True)
    p.add_argument("--night", required=True)

    p = sub.add_parser("factor", help="compute and save the correction factor")
    p.add_argument("--day", required=True)
    p.add_argument("--night", required=True)
    p.add_argument("--from", dest="from_state", default="night", choices=["day", "night"])
    p.add_argument("--to", dest="to_state", default="day", choices=["day", "night"])
    p.add_argument("--weights", default="counts", choices=["counts", "mean"])
    p.add_argument("--out", required=True)

    p = sub.add_parser("apply", help="correct a beam HDF5 file")
    p.add_argument("--factor", required=True)
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--start", default=None, help="ISO UTC start time (inclusive)")
    p.add_argument("--stop", default=None, help="ISO UTC stop time (exclusive)")
    p.add_argument("--planes", nargs="+", default=None)
    p.add_argument("--overwrite", action="store_true")

    sub.add_parser("selftest", help="run a quick synthetic check")

    args = ap.parse_args(argv)

    if args.cmd == "info":
        from .settings import SettingsPair
        s = SettingsPair(args.day, args.night)
        print(json.dumps(s.summary(), indent=2, default=str))
        return 0

    if args.cmd == "factor":
        from .settings import SettingsPair
        s = SettingsPair(args.day, args.night)
        f = s.factor(to_state=args.to_state, from_state=args.from_state, weights=args.weights)
        f.save(args.out)
        import numpy as np
        anchors = [20e6, 25e6, 28.3e6, 30e6, 40e6, 50e6, 60e6, 70e6, 80e6, 85e6]
        print(f"saved {args.out}: {args.from_state}->{args.to_state}, "
              f"{f.r_volt.size} EQ channels, spacing {f.spacing_hz:.5f} Hz, offset {f.eq_offset}")
        for a in anchors:
            c = f.eval_power([a])[0]
            print(f"  {a/1e6:7.2f} MHz: C = {c:.5f} ({10*np.log10(c):+.3f} dB)")
        return 0

    if args.cmd == "apply":
        from .apply import correct_hdf5, PLANES
        from .settings import FrequencyFactor
        f = FrequencyFactor.load(args.factor)
        t0 = _iso(args.start) if args.start else None
        t1 = _iso(args.stop) if args.stop else None
        info = correct_hdf5(args.input, args.output, f,
                            planes=tuple(args.planes) if args.planes else PLANES,
                            t0=t0, t1=t1, overwrite=args.overwrite)
        print(json.dumps(info, indent=2))
        return 0

    if args.cmd == "selftest":
        import numpy as np
        from .settings import FrequencyFactor, EQ_SPACING_HZ, BEAM_EQ_OFFSET
        r = np.linspace(0.8, 1.2, 1186)
        f = FrequencyFactor(r_volt=np.concatenate([r, np.ones(4096 - 1186)]))
        freq = (np.arange(3) + BEAM_EQ_OFFSET) * EQ_SPACING_HZ + np.array([0.0, 0.1, -0.1])
        c = f.eval_power(freq)
        assert np.allclose(c, r[BEAM_EQ_OFFSET:BEAM_EQ_OFFSET + 3] ** 2)
        assert np.array_equal(f.eq_channel([13398437.5]), [BEAM_EQ_OFFSET])
        try:
            f.eval_power([110e6])
            raise AssertionError("expected range error")
        except ValueError:
            pass
        print("selftest ok")
        return 0

    return 2


if __name__ == "__main__":
    sys.exit(main())
