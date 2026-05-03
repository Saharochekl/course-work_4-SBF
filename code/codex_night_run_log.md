# Codex Night Run Log

This file is populated by `code/night_sbf_fast_experiments.py`.
### exp01_baseline_current_k
- Timestamp: 2026-05-02 05:07:13
- Model variant: `baseline`
- Main k-window: `(0.03, 0.4)`
- Region k-windows: `[(0.02, 0.35), (0.03, 0.4)]`
- FFT E(k) realizations: main=`4`, region=`2`
- Decision: `keep`
- Why: baseline reference
- Science numbers:
  elliptical corrected mbar = `27.1099`
  circular inner corrected mbar = `27.3643`
  circular outer corrected mbar = `27.1804`
  weighted corrected mbar = `27.3164`
  formal sigma = `0.0526`
  annulus scatter = `0.0920`
  adopted sigma = `0.0920`
- Residual diagnostics:
  elliptical_chosen: median=`1.0215e-01`, mean=`1.5518e-01`, sign=`positive`
  circular_inner_lit: median=`6.8445e-01`, mean=`7.4966e-01`, sign=`positive`
  circular_outer_lit: median=`-1.7043e-01`, mean=`-9.8963e-02`, sign=`negative`
- zero-cross circle px = `587.26`
- zero-cross ellipse px = `834.95`

### exp02_conservative_lowk_cut
- Timestamp: 2026-05-02 05:11:39
- Model variant: `baseline`
- Main k-window: `(0.05, 0.35)`
- Region k-windows: `[(0.04, 0.3), (0.05, 0.35)]`
- FFT E(k) realizations: main=`4`, region=`2`
- Decision: `reject`
- Why: same residual image as baseline; annulus consistency worsened
- Science numbers:
  elliptical corrected mbar = `26.6734`
  circular inner corrected mbar = `27.3650`
  circular outer corrected mbar = `26.9854`
  weighted corrected mbar = `27.1800`
  formal sigma = `0.1695`
  annulus scatter = `0.1898`
  adopted sigma = `0.1898`
- Residual diagnostics:
  elliptical_chosen: median=`1.0215e-01`, mean=`1.5518e-01`, sign=`positive`
  circular_inner_lit: median=`6.8445e-01`, mean=`7.4966e-01`, sign=`positive`
  circular_outer_lit: median=`-1.7043e-01`, mean=`-9.8963e-02`, sign=`negative`
- zero-cross circle px = `587.26`
- zero-cross ellipse px = `834.95`

### exp03_conservative_highk_cut
- Timestamp: 2026-05-02 05:15:45
- Model variant: `baseline`
- Main k-window: `(0.03, 0.25)`
- Region k-windows: `[(0.02, 0.2), (0.03, 0.25)]`
- FFT E(k) realizations: main=`4`, region=`2`
- Decision: `keep`
- Why: same residual image as baseline; annulus consistency improved
- Science numbers:
  elliptical corrected mbar = `27.2710`
  circular inner corrected mbar = `27.4778`
  circular outer corrected mbar = `27.3796`
  weighted corrected mbar = `27.4585`
  formal sigma = `0.0574`
  annulus scatter = `0.0491`
  adopted sigma = `0.0574`
- Residual diagnostics:
  elliptical_chosen: median=`1.0215e-01`, mean=`1.5518e-01`, sign=`positive`
  circular_inner_lit: median=`6.8445e-01`, mean=`7.4966e-01`, sign=`positive`
  circular_outer_lit: median=`-1.7043e-01`, mean=`-9.8963e-02`, sign=`negative`
- zero-cross circle px = `587.26`
- zero-cross ellipse px = `834.95`

### exp04_conservative_both_cuts
- Timestamp: 2026-05-02 05:19:52
- Model variant: `baseline`
- Main k-window: `(0.05, 0.25)`
- Region k-windows: `[(0.04, 0.2), (0.05, 0.25)]`
- FFT E(k) realizations: main=`4`, region=`2`
- Decision: `reject`
- Why: same residual image as baseline; annulus consistency worsened
- Science numbers:
  elliptical corrected mbar = `26.8514`
  circular inner corrected mbar = `27.5635`
  circular outer corrected mbar = `27.2131`
  weighted corrected mbar = `27.3941`
  formal sigma = `0.1961`
  annulus scatter = `0.1752`
  adopted sigma = `0.1961`
- Residual diagnostics:
  elliptical_chosen: median=`1.0215e-01`, mean=`1.5518e-01`, sign=`positive`
  circular_inner_lit: median=`6.8445e-01`, mean=`7.4966e-01`, sign=`positive`
  circular_outer_lit: median=`-1.7043e-01`, mean=`-9.8963e-02`, sign=`negative`
- zero-cross circle px = `587.26`
- zero-cross ellipse px = `834.95`

### exp05_literature_closer_lowk
- Timestamp: 2026-05-02 05:23:57
- Model variant: `baseline`
- Main k-window: `(0.01, 0.25)`
- Region k-windows: `[(0.005, 0.2), (0.01, 0.25)]`
- FFT E(k) realizations: main=`4`, region=`2`
- Decision: `keep`
- Why: same residual image as baseline; annulus consistency improved
- Science numbers:
  elliptical corrected mbar = `27.0973`
  circular inner corrected mbar = `27.1785`
  circular outer corrected mbar = `27.1908`
  weighted corrected mbar = `27.1857`
  formal sigma = `0.0411`
  annulus scatter = `0.0062`
  adopted sigma = `0.0411`
- Residual diagnostics:
  elliptical_chosen: median=`1.0215e-01`, mean=`1.5518e-01`, sign=`positive`
  circular_inner_lit: median=`6.8445e-01`, mean=`7.4966e-01`, sign=`positive`
  circular_outer_lit: median=`-1.7043e-01`, mean=`-9.8963e-02`, sign=`negative`
- zero-cross circle px = `587.26`
- zero-cross ellipse px = `834.95`

### exp06_fitted_splice_best_k
- Timestamp: 2026-05-02 05:28:42
- Model variant: `fitted_splice`
- Main k-window: `(0.01, 0.25)`
- Region k-windows: `[(0.005, 0.2), (0.01, 0.25)]`
- FFT E(k) realizations: main=`4`, region=`2`
- Decision: `reject`
- Why: residual bias improved, but annulus scatter/formal error became too poor
- Science numbers:
  elliptical corrected mbar = `27.5100`
  circular inner corrected mbar = `27.5847`
  circular outer corrected mbar = `27.2829`
  weighted corrected mbar = `27.4632`
  formal sigma = `0.0407`
  annulus scatter = `0.1509`
  adopted sigma = `0.1509`
- Residual diagnostics:
  elliptical_chosen: median=`-5.8628e-02`, mean=`-2.7825e-03`, sign=`negative`
  circular_inner_lit: median=`-6.4397e-02`, mean=`-1.2298e-02`, sign=`negative`
  circular_outer_lit: median=`-1.8057e-01`, mean=`-1.1491e-01`, sign=`negative`
- zero-cross circle px = `159.11`
- zero-cross ellipse px = `196.68`

