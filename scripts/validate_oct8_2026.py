#!/usr/bin/env python3
"""Second-pass diagnostics: multi-day validation, ripple metric, settings history, figures."""
import os
import numpy as np
import scipy.io as sio
from scipy.ndimage import median_filter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = '/data1/pzhang/dev/lwaspec_setting_corr'
S = 1.7782794100389228
COUNTS = np.array([124, 42, 64, 130, 96, 30, 218], float)
COUNTS = COUNTS / COUNTS.sum()

d = sio.loadmat(f'{BASE}/settings/20251120a-settingsAll-day-FW7p5.mat', squeeze_me=True)
n = sio.loadmat(f'{BASE}/settings/20251120f-settingsAll-night-FW7p5.mat', squeeze_me=True)
coefD = np.asarray(d['coef'], float)
coefN = np.asarray(n['coef'], float)
Rg = np.asarray(coefD, float) / np.asarray(coefN, float)          # raw day/night voltage ratio per group
Rg_norm = Rg / S                                                   # structured residual ~1
R_all = (Rg * COUNTS[:, None]).sum(0)                              # count-weighted mean raw ratio
C_lin = R_all ** 2                                                 # power correction: (weighted-mean ratio)^2
C_g0 = Rg[0] ** 2
R2_sq = (Rg_norm ** 2 * COUNTS[:, None]).sum(0)                    # power-weighted mean of ratio^2
C_sq = (R2_sq) * (S ** 2)
print('group spread (norm R) below/above eq1186:')
for g in range(7):
    lo = Rg_norm[g][:1186]
    hi = Rg_norm[g][1186:]
    print(f'  grp{g}: lo med={np.median(lo):.4f} std={lo.std():.4f} | hi med={np.median(hi):.4f} std={hi.std():.4f}')

eqidx = 560 + np.arange(3072)
Cn = {k: v[eqidx] for k, v in (('lin', C_lin), ('g0', C_g0), ('sq', C_sq))}
print('C variants max |dB diff| vs lin:',
      {k: round(float(np.max(np.abs(10 * np.log10(v / Cn['lin'])))), 3) for k, v in Cn.items()})

freq = None
z = np.load(f'{BASE}/out/oct8_a.npz')
freq = z['freq']
m_hi = (freq > 29e6) & (freq < 86.5e6)
m_lo = (freq > 14e6) & (freq < 28.0e6)


def ripple(x):
    sel = m_hi & np.isfinite(x) & (x > 0)
    lg = np.log10(x[sel])
    lg = np.interp(np.arange(lg.size), np.where(np.isfinite(lg))[0],
                   lg[np.isfinite(lg)]) if (~np.isfinite(lg)).any() else lg
    return float((lg - median_filter(lg, 101)).std())


def load(fn):
    return np.load(f'{BASE}/out/{fn}.npz') if os.path.exists(f'{BASE}/out/{fn}.npz') else None


def getz(zx, key):
    return zx[key] if zx is not None and key in zx.files else None


z7 = load('oct7_a')
zd = load('oct8_day')
z8b = load('oct8_b')
z7b = load('oct7_b')
z6 = load('oct6_a')
z8c = load('oct8_c')
z8c2 = load('oct8_c2')

s8 = getz(z, 'XX_20261008T152000')
s7 = getz(z7, 'XX_20261007T152000')
s6 = getz(z6, 'XX_20261006T152000')
sd = getz(zd, 'XX_20261008T233000')
s8b = getz(z8b, 'XX_20261008T163000')
s7b = getz(z7b, 'XX_20261007T163000')
s8c1 = getz(z8c, 'XX_20261008T203000')
s8c2 = getz(z8c2, 'XX_20261008T213000')


def resid(num, den, C):
    r = (num / den) / C
    ok = np.isfinite(r) & (r > 0)
    out = {}
    for mk, nm in ((m_hi & ok, 'hi'), (m_lo & ok, 'lo')):
        db = 10 * np.log10(r[mk])
        out[nm] = (np.median(db), db.std(), int(mk.sum()))
    return out


print('residuals (median, std, n) of day/night / C, C=lin:')
pairs = [('oct7/oct8 15:20', s7, s8)]
if s6 is not None:
    pairs.append(('oct6/oct8 15:20', s6, s8))
for name, num, den in pairs:
    r0 = resid(num, den, Cn['lin'])
    r1 = resid(num, den, Cn['g0'])
    r2 = resid(num, den, Cn['sq'])
    print(f'  {name}: lin hi={r0["hi"]} lo={r0["lo"]}')
    print(f'       g0  hi={r1["hi"]}')
    print(f'       sq  hi={r2["hi"]}')

print()
print('ripple metric (std of log10 minus medfilt101, 29-86.5 MHz):')
print('  oct8 15:20 raw      :', round(ripple(s8), 4))
print('  oct8 15:20 x C      :', round(ripple(s8 * Cn['lin']), 4))
print('  oct7 15:20 (day ref):', round(ripple(s7), 4))
if s6 is not None:
    print('  oct6 15:20 (day ref):', round(ripple(s6), 4))
print('  oct8 16:30 raw      :', round(ripple(s8b), 4))
print('  oct8 16:30 x C      :', round(ripple(s8b * Cn['lin']), 4))
print('  oct8 23:30 (day)    :', round(ripple(sd), 4))
print('  C ripple            :', round(ripple(Cn['lin']), 4))

print()
print('C power correction (dB) at anchor freqs:')
for f0 in (20e6, 25e6, 28e6, 30e6, 35e6, 40e6, 50e6, 60e6, 70e6, 80e6, 85e6):
    i = int(np.argmin(np.abs(freq - f0)))
    print(f'  {freq[i]/1e6:7.2f} MHz: {10*np.log10(Cn["lin"][i]):+.3f} dB')

print()
print('within-day oct8 solar change 16:30/15:20 (same state, no settings diff):')
rr = s8b / s8
ok = np.isfinite(rr) & (rr > 0)
print('  hi med=%.2f dB' % np.median(10 * np.log10(rr[m_hi & ok])))
print('oct8 continuity toward evening (raw medians / corrected):')
for tt, vv in (('20:30', s8c1), ('21:30', s8c2)):
    if vv is None:
        continue
    okv = m_hi & np.isfinite(vv) & (vv > 0)
    print(f'  {tt}: raw hi-med={np.median(vv[okv]):.4g}  xC={np.median((vv*Cn["lin"])[okv]):.4g}')

print()
print('older settings pair check (20250612):')
old_d = f'{BASE}/settings/old/20250612-settingsAll-day.mat'
old_n = f'{BASE}/settings/old/20250612-settingsAll-night.mat'
if os.path.exists(old_d) and os.path.exists(old_n):
    dd = sio.loadmat(old_d, squeeze_me=True)
    nn = sio.loadmat(old_n, squeeze_me=True)
    if 'coef' in dd and 'coef' in nn:
        cd = np.asarray(dd['coef'], float)
        cn2 = np.asarray(nn['coef'], float)
        rr2 = cd / cn2
        print('  coef ratio med per group:', [round(float(np.median(r)), 5) for r in rr2])
        print('  ratio std below/above 1186 (grp0):', round(float(rr2[0][:1186].std()), 5),
              round(float(rr2[0][1186:].std()), 5))
        if 'settings' in dd and 'settings' in nn:
            a1 = np.asarray(dd['settings'], float)[..., 0]
            a2 = np.asarray(nn['settings'], float)[..., 0]
            print('  AT1 diff uniq:', np.unique(np.round(a2 - a1, 4))[:10])
            b1 = np.asarray(dd['settings'], float)[..., 1]
            b2 = np.asarray(nn['settings'], float)[..., 1]
            print('  AT2 identical:', np.array_equal(b1, b2), ' AT2 diff uniq:', np.unique(np.round(b2 - b1, 4))[:10])
            c1 = np.asarray(dd['settings'], float)[..., 2]
            c2 = np.asarray(nn['settings'], float)[..., 2]
            print('  filter identical:', np.array_equal(c1, c2))
            e1 = np.asarray(dd['settings'], float)[..., 3]
            e2 = np.asarray(nn['settings'], float)[..., 3]
            print('  feePower identical:', np.array_equal(e1, e2))
    else:
        print('  keys day:', sorted(k for k in dd if not k.startswith('__'))[:20])
else:
    print('  (old pair not fetched)')

print()
print('save factor')
np.savez(f'{BASE}/out/factor_v1.npz', eq_chan=np.arange(4096), R_all=R_all, C_lin=C_lin,
         R_group=Rg_norm, counts=COUNTS, freq_eq_MHz=np.arange(4096) * 23925.78125 / 1e6)

fig, ax = plt.subplots(3, 2, figsize=(15, 11))
ax[0, 0].plot(freq / 1e6, np.log10(s8), lw=.7, label='oct8 15:20 raw (night)')
ax[0, 0].plot(freq / 1e6, np.log10(s8 * Cn['lin']), lw=.7, label='oct8 15:20 x C')
ax[0, 0].plot(freq / 1e6, np.log10(s7), lw=.7, alpha=.6, label='oct7 15:20 (day)')
if s6 is not None:
    ax[0, 0].plot(freq / 1e6, np.log10(s6), lw=.7, alpha=.6, label='oct6 15:20 (day)')
ax[0, 0].legend(fontsize=7); ax[0, 0].set_ylabel('log10 power'); ax[0, 0].set_title('XX spectra')
ax[0, 1].plot(freq / 1e6, 10 * np.log10(Cn['lin']), lw=.8, label='C (dB)')
ax[0, 1].plot(freq / 1e6, 10 * np.log10(s7 / s8), lw=.5, alpha=.6, label='r1 oct7/oct8')
if s6 is not None:
    ax[0, 1].plot(freq / 1e6, 10 * np.log10(s6 / s8), lw=.5, alpha=.6, label='r oct6/oct8')
ax[0, 1].legend(fontsize=7); ax[0, 1].set_title('correction C vs measured ratios (dB)')
ax[1, 0].plot(freq / 1e6, 10 * np.log10((s7 / s8) / Cn['lin']), lw=.6, label='oct7')
if s6 is not None:
    ax[1, 0].plot(freq / 1e6, 10 * np.log10((s6 / s8) / Cn['lin']), lw=.6, alpha=.7, label='oct6')
ax[1, 0].axhline(0, color='k', lw=.5); ax[1, 0].legend(fontsize=7)
ax[1, 0].set_title('residual / C (dB)'); ax[1, 0].set_ylim(-3, 3)
ax[1, 1].plot(freq / 1e6, np.log10(s8), lw=.7); ax[1, 1].plot(freq / 1e6, np.log10(s8 * Cn['lin']), lw=.7)
ax[1, 1].set_xlim(28, 45); ax[1, 1].set_title('zoom 28-45 MHz raw vs xC')
ax[2, 0].plot(freq / 1e6, np.log10(s8), lw=.7); ax[2, 0].plot(freq / 1e6, np.log10(s8 * Cn['lin']), lw=.7)
ax[2, 0].plot(freq / 1e6, np.log10(s7), lw=.7, alpha=.6)
ax[2, 0].set_xlim(45, 70); ax[2, 0].set_title('zoom 45-70 MHz (also oct7)')
ax[2, 1].plot(freq / 1e6, np.log10(s8 * Cn['lin']), lw=.7, label='oct8 15:20 x C')
ax[2, 1].plot(freq / 1e6, np.log10(s7), lw=.7, alpha=.7, label='oct7 15:20')
ax[2, 1].set_xlim(70, 87); ax[2, 1].legend(fontsize=7); ax[2, 1].set_title('zoom 70-87 MHz')
for a in ax.ravel():
    a.set_xlabel('MHz'); a.grid(alpha=.3)
plt.tight_layout()
plt.savefig(f'{BASE}/out/diag_secondlook.png', dpi=130)
print('fig saved', f'{BASE}/out/diag_secondlook.png')
