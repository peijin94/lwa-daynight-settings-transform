#!/usr/bin/env python3
"""Check AT1 uniformity and EQ residual structure R(nu) = coefD/coefN / 1.7783."""
import sys
import numpy as np
import scipy.io as sio

day = sio.loadmat(sys.argv[1], squeeze_me=True)
night = sio.loadmat(sys.argv[2], squeeze_me=True)

ad = np.asarray(day['settings'])[..., 0]
an = np.asarray(night['settings'])[..., 0]
print('AT1 day uniq :', np.unique(ad))
print('AT1 night uniq:', np.unique(an))
print('AT1 diff uniq (night-day):', np.unique(np.round(an - ad, 6)))
ad2 = np.asarray(day['settings'])[..., 1]
an2 = np.asarray(night['settings'])[..., 1]
print('AT2 identical:', np.array_equal(ad2, an2), ' day uniq sample:', np.unique(ad2)[:5], '... n uniq', np.unique(ad2).size)

cd = np.asarray(day['coef'], dtype=float)
cn = np.asarray(night['coef'], dtype=float)
S = 1.7782794100389228  # 10^(5/20)
R = (cd / cn) / S
print()
print('R(nu) = (coef_day/coef_night)/1.7783 per group:')
for g in range(R.shape[0]):
    r = R[g]
    print(f'  grp{g}: min={r.min():.4f} med={np.median(r):.4f} max={r.max():.4f} std={r.std():.4f} '
          f'p5={np.percentile(r,5):.4f} p95={np.percentile(r,95):.4f}')

print()
print('Structure around ch1186 (the night comment: "ignoring subbands starting at ch1186"):')
for g in range(R.shape[0]):
    r = R[g]
    lo = r[:1186]
    hi = r[1186:]
    print(f'  grp{g}: R<1186 med={np.median(lo):.4f} std={lo.std():.4f} | R>=1186 med={np.median(hi):.4f} std={hi.std():.4f}')

print()
print('R sample every 256 channels (grp0..6):')
for g in range(R.shape[0]):
    vals = ','.join('%.3f' % v for v in R[g][::256])
    print(f'  grp{g}: {vals}')

print()
print('power-domain deviation dP = R^2-1 (percent), per group:')
for g in range(R.shape[0]):
    d = R[g] ** 2 - 1
    print(f'  grp{g}: min={100*d.min():.1f}% med={100*np.median(d):.1f}% max={100*d.max():.1f}%')

# group sizes and how many live beam antennas per group (rough: all 704)
sizes = [np.atleast_1d(day[f'eq{g}']).size for g in range(7)]
print()
print('group sizes:', sizes, 'sum', sum(sizes))
print('day time:', day['time'], ' night time:', night['time'])
print('comment day  :', day['comment'])
print('comment night:', night['comment'])
