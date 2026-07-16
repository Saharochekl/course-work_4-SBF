# SBF-3 other-filter distance validation

Run date: 2026-07-16.

## Question and design

This is a one-anchor/one-validation differential test of the generalized JWST
pipeline with F356W as the SBF signal image and F277W as the color image.
NGC 1380 defines a provisional F356W zero point from its independent TRGB
distance. That zero point is then applied, without refitting, to NGC 1404 and
compared with the NGC 1404 TRGB distance.

Both runs used the unchanged `sbf-3.ipynb` with SHA256
`49183713005966ed969528953019f057ed3440da8b81f444e7eba1af00df2390`.
The inputs are public JWST/NIRCam calibrated i2d SCI images in MJy/sr. The two
filter grids were deliberately not treated as pixel-aligned: color values were
sampled through WCS with bilinear interpolation.

## Measurements

| Quantity | NGC 1380 (anchor) | NGC 1404 (validation) |
|---|---:|---:|
| recommended mbar(F356W) | 27.64022 +/- 0.09115 | 27.86951 +/- 0.17491 |
| inner-annulus mbar(F356W) | 27.74914 +/- 0.04126 | 27.88308 +/- 0.08625 |
| outer-annulus mbar(F356W) | 27.56684 +/- 0.03387 | 27.53326 +/- 0.42928 |
| F277W-F356W | -0.243060 | -0.233659 |
| adopted TRGB modulus | 31.408 +/- 0.025 | 31.366 +/- 0.025 |

The NGC 1380 anchor gives

`Mbar(F356W) = -3.76778 +/- 0.09452 mag`.

Applied to NGC 1404, this predicts

`mu = 31.63728 +/- 0.19882 mag`, or `D = 21.25 Mpc`.

The independent NGC 1404 TRGB value is `mu = 31.366 +/- 0.025`, or
`D = 18.76 Mpc`. The residual is therefore

`+0.27128 +/- 0.20038 mag = 1.35 sigma`, corresponding to a central distance
offset of `+13.31%`.

The quoted differential uncertainty uses the two random TRGB errors. A common
TRGB zero-point systematic is not added twice because it cancels between the
anchor and validation galaxy; it would matter for an absolute F356W zero point.

Thus the result is statistically consistent only because the SBF uncertainty
is large. It is not a precision validation that distance is invariant under a
filter change.

## Diagnostics

For NGC 1404, the weighted results at kmin 0.01, 0.03 and 0.04 are 27.64247,
27.91419 and 27.86951 mag. The outer annulus is strongly low-k sensitive and
its main-window value differs from the inner annulus by 0.350 mag. Their
F277W-F356W colors differ by only about 0.00125 mag, so the annulus discrepancy
is not explained by this color gradient. Large-scale model residuals are the
more likely cause.

The adopted 0.175-mag uncertainty includes half the inner/outer difference,
but the notebook still reports that a separate pipeline systematic has not
been measured. The matched inner-annulus-only differential residual is
`+0.17594 +/- 0.10194 mag` (1.73 sigma; +8.44% in distance), so discarding the
outer annulus does not make the population/calibration term vanish.

Across the two galaxies, F277W-F356W changes by only 0.00940 mag. Explaining
the weighted SBF difference with two points would require an ill-conditioned
slope near 28.9 mag/mag. This color baseline is too short for a useful
two-object calibration. The pre-existing F150W results already show a
0.159-mag intrinsic SBF difference between these galaxies after removing the
TRGB distance difference, which independently demonstrates a stellar-
population term.

## Verdict

- The generalized input/WCS/PSF/SBF path works end to end for F356W+F277W.
- An F150W zero point cannot be reused in F356W.
- A one-galaxy F356W zero point produces a 0.27-mag central residual on the
  second galaxy; current uncertainty makes this only a 1.35-sigma discrepancy.
- F277W-F356W has inadequate color leverage here. A wider baseline such as
  F150W-F356W or an optical-to-IR color is the rational calibration variable.
- GO-5989 contains the useful F150W/F356W image pair. Its current operational
  manifest measures SBF in F150W and uses F356W-F150W as color; an F356W SBF
  experiment must explicitly swap those signal/color roles and retain a
  separate calibration family.

This test supports filter-generic execution, not filter-independent distance
calibration. More independent-distance galaxies and explicit pipeline variants
are required before claiming a calibrated F356W distance scale.

References used for the interpretation: Jensen et al. 2015 near-IR SBF color
relations (arXiv:1505.00400), TRGB-SBF Project I (arXiv:2405.03743), and the
local Project IV TRGB table (arXiv:2603.11160).
