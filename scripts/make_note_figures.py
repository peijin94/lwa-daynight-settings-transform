#!/usr/bin/env python3
"""Generate the three note figures from out/*.npz + settings pair."""
import numpy as np
import scipy.io as sio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = '/data1/pzhang/dev/lwaspec_setting_corr'
S = 1.7782794100389228
COUNTS = np.array([124, 42, 64, 130, 96, 30, 218], float)
COUNTS = COUNTS / COUNTS.sum()
SPACING = 23925.78125

d = sio.loadmat(f'{BASE}/settings/20251120a-settingsAll-day-FW7p5.mat', squeeze_me=True)
n = sio.loadmat(f'{BASE}/settings/20251120f-settingsAll-night-FW7p5.mat', squeeze_me=True)
Rg = np.asarray(d['coef'], float) / np.asarray(n['coef'], float)
R_all = (Rg * COUNTS[:, None]).sum(0)
C = R_all ** 2
freq_eq = np.arange(4096) * SPACING / 1e6

z8 = np.load(f'{BASE}/out/oct8_a.npz'); z7 = np.load(f'{BASE}/out/oct7_a.npz')
z6 = np.load(f'{BASE}/out/oct6_a.npz'); zd = np.load(f'{BASE}/out/oct8_day.npz')
z8b = np.load(f'{BASE}/out/oct8_b.npz'); z7b = np.load(f'{BASE}/out/oct7_b.npz')
z8c = np.load(f'{BASE}/out/oct8_c.npz'); z7c = np.load(f'{BASE}/out/oct7_c.npz')
z7c2 = np.load(f'{BASE}/out/oct7_c2.npz')

freq = z8['freq']
eqidx = 560 + np.arange(freq.size)
Cf = C[eqidx]
Rm = R_all[eqidx]
FMHz = freq / 1e6

s8 = z8['XX_20261008T152000']; s7 = z7['XX_20261007T152000']; s6 = z6['XX_20261006T152000']
sd = zd['XX_20261008T233000']; s8b = z8b['XX_20261008T163000']; s7b = z7b['XX_20261007T163000']
s8c = z8c['XX_20261008T203000']; s7c = z7c['XX_20261007T203000']; s7d = z7c2['XX_20261007T233000']

# ---------- fig_factor ----------
fig, ax = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
ax[0].plot(FMHz, 10 * np.log10(Cf), lw=1.2, color='C3')
ax[0].axvline(28.38, color='k', ls=':', lw=1)
ax[0].text(28.6, 5.15, '28.38 MHz\n(EQ ch 1186)', fontsize=8)
ax[0].set_ylabel('C(f)  [dB, power]')
ax[0].set_title('night->day power correction from the 20251120 FW7p5 settings pair')
ax[0].grid(alpha=.3)
for g in range(7):
    ax[1].plot(freq_eq, Rg[g] * 1.0, lw=.8, label=f'eq{g}')
ax[1].set_xlabel('MHz'); ax[1].set_ylabel('coef_day / coef_night  [voltage]')
ax[1].legend(fontsize=7, ncol=4)
ax[1].grid(alpha=.3)
plt.tight_layout(); plt.savefig(f'{BASE}/out/fig_factor.png', dpi=150); plt.close()

# ---------- fig_acceptance ----------
fig, ax = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
ax[0].plot(FMHz, np.log10(s8), lw=.6, color='C0', label='2026-10-08 15:20 raw (night settings)')
ax[0].plot(FMHz, np.log10(s8 * Cf), lw=.6, color='C1', label='2026-10-08 15:20 x C(f) corrected')
ax[0].plot(FMHz, np.log10(s7), lw=.6, color='C2', alpha=.8, label='2026-10-07 15:20 (day settings)')
ax[0].plot(FMHz, np.log10(s6), lw=.6, color='C4', alpha=.6, label='2026-10-06 15:20 (day settings)')
ax[0].set_ylabel('log10 power'); ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)
ax[0].set_title('XX beam spectra, same solar elevation (15:20 UTC)')
ax[1].plot(FMHz, 10 * np.log10(s7 / s8), lw=.5, color='C2', label='day/night ratio  Oct7/Oct8  raw')
ax[1].plot(FMHz, 10 * np.log10(s6 / s8), lw=.5, color='C4', alpha=.6, label='Oct6/Oct8 raw')
ax[1].plot(FMHz, 10 * np.log10((s7 / s8) / Cf), lw=.5, color='C1', label='Oct7/Oct8 / C(f)  residual')
ax[1].axhline(0, color='k', lw=.4)
for y in (5,): ax[1].axhline(y, color='k', lw=.4, ls=':')
ax[1].set_ylim(-6, 12); ax[1].set_xlabel('MHz'); ax[1].set_ylabel('dB')
ax[1].legend(fontsize=8, ncol=3); ax[1].grid(alpha=.3)
plt.tight_layout(); plt.savefig(f'{BASE}/out/fig_acceptance.png', dpi=150); plt.close()

# ---------- fig_sequence ----------
def bandmed(x):
    m = (freq > 35e6) & (freq < 85e6) & np.isfinite(x) & (x > 0)
    return np.median(x[m])

times = ['15:20', '16:30', '20:30', '23:30']
o7 = [bandmed(v) for v in (s7, s7b, s7c, s7d)]
o8_raw = [bandmed(v) for v in (s8, s8b, s8c, sd)]
o8_cor = [bandmed(v * Cf) for v in (s8, s8b, s8c, sd)]  # Cf(=1 for f<28.38; for 23:30 it's day data but still apply note: not right)

# careful: for 23:30 the data is already day-state; applying Cf would be wrong. Correct only the night-state ones.
o8_cor = [bandmed(s8 * Cf), bandmed(s8b * Cf), bandmed(s8c * Cf), bandmed(sd)]

x = np.arange(len(times))
w = 0.27
ref = o7[0]
fig, ax = plt.subplots(figsize=(9, 5))
ax.bar(x - w, 10 * np.log10(np.array(o7) / ref), w, label='Oct 7 (day settings, ref)')
ax.bar(x, 10 * np.log10(np.array(o8_raw) / ref), w, label='Oct 8 raw')
ax.bar(x + w, 10 * np.log10(np.array(o8_cor) / ref), w, label='Oct 8 corrected (night->day)')
ax.set_xticks(x); ax.set_xticklabels(times)
ax.set_ylabel('median level 35-85 MHz [dB re Oct7 15:20]')
ax.axhline(0, color='k', lw=.4)
ax.legend(fontsize=8)
ax.set_title('oct8 15:20/16:30/20:30 recorded in night-EQ state (correction applied); 23:30 already day')
ax.grid(alpha=.3, axis='y')
plt.tight_layout(); plt.savefig(f'{BASE}/out/fig_sequence.png', dpi=150); plt.close()

print('figs saved: fig_factor.png fig_acceptance.png fig_sequence.png')
print('level check dB (raw oct8 - oct7) per time:', [round(10*np.log10(a/b),2) for a,b in zip(o8_raw,o7)])
print('level check dB (corrected - oct7):', [round(10*np.log10(a/b),2) for a,b in zip(o8_cor,o7)])
