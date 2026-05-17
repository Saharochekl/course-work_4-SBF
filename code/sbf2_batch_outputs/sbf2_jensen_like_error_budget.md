# Fast Jensen-like SBF Error Budget

No isophote/model/PSF/SBF-region rerun was performed.

Formula:

```text
sigma_fast^2 = sigma_Pk^2 + sigma_bkg^2 + sigma_PSF^2 + sigma_Pr^2 + sigma_ext^2
```

- `sigma_Pk` uses Jensen-style weighted average of the inner/outer annulus errors already measured by sbf-2.
- `sigma_bkg` is auto-propagated from saved `[BKG-CHECK]` sky residual and annulus `Imean`, unless supplied by CLI.
- `sigma_PSF` is read from saved systematics branch CSVs when available; otherwise DET_DIST-vs-DET_SAMP is measured on saved residual FITS.
- `sigma_Pr` uses proxy `1.0857 * 0.25 * (Pr/P0) / (1 - Pr/P0)`.
- `sigma_ext` is direct CLI, CSV, or automatic IRSA/SFD `sigma_E(B-V)` propagated with a near-IR CCM F150W coefficient.

| galaxy | mbar_150 | sigma_Pk | sigma_bkg | sigma_PSF | sigma_Pr | sigma_ext | sigma_fast | missing | bkg_status | psf_status | ext_status | flags |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|---|
| NGC 1380 | 27.8871 | 0.0370 | 0.0004 | 0.0072 | 0.0067 | 0.0004 | 0.0382 |  | auto_from_median_bkg_check_and_Imean_included | auto_from_saved_psf_branch_included | auto_irsa_sigma_ebv_included |  |
| NGC 1399 | 28.1249 | 0.0972 | 0.0029 | 0.0071 | 0.0027 | 0.0003 | 0.0976 |  | auto_from_median_bkg_check_and_Imean_included | auto_from_fast_detdist_detsamp_rerun_included | auto_irsa_sigma_ebv_included |  |
| NGC 1404 | 28.0042 | 0.0386 | 0.0000 | 0.0072 | 0.0025 | 0.0002 | 0.0393 |  | auto_from_median_bkg_check_and_Imean_included | auto_from_fast_detdist_detsamp_rerun_included | auto_irsa_sigma_ebv_included |  |
| NGC 4472 | 27.5126 | 0.0338 | 0.0009 | 0.0074 | 0.0097 | 0.0001 | 0.0359 |  | auto_from_median_bkg_check_and_Imean_included | auto_from_fast_detdist_detsamp_rerun_included | auto_irsa_sigma_ebv_included |  |
| NGC 4552 | 27.6180 | 0.0322 | 0.0001 | 0.0073 | 0.0060 | 0.0010 | 0.0335 |  | auto_from_median_bkg_check_and_Imean_included | auto_from_fast_detdist_detsamp_rerun_included | auto_irsa_sigma_ebv_included |  |
| NGC 4636 | 27.5483 | 0.0432 | 0.0002 | 0.0073 | 0.0086 | 0.0004 | 0.0446 |  | auto_from_median_bkg_check_and_Imean_included | auto_from_fast_detdist_detsamp_rerun_included | auto_irsa_sigma_ebv_included |  |
| NGC 4649 | 27.5694 | 0.0380 | 0.0009 | 0.0072 | 0.0059 | 0.0004 | 0.0391 |  | auto_from_median_bkg_check_and_Imean_included | auto_from_fast_detdist_detsamp_rerun_included | auto_irsa_sigma_ebv_included |  |
| NGC 4697 | 26.7409 | 0.0422 | 0.0012 | 0.0071 | 0.0058 | 0.0012 | 0.0432 |  | auto_from_median_bkg_check_and_Imean_included | auto_from_fast_detdist_detsamp_rerun_included | auto_irsa_sigma_ebv_included | isophote_real_only_failed;sersic_fit_warning |
| NGC 4486 | 27.5997 | 0.0451 | 0.0002 | 0.0068 | 0.0033 | 0.0004 | 0.0458 |  | auto_from_median_bkg_check_and_Imean_included | auto_from_fast_detdist_detsamp_rerun_included | auto_irsa_sigma_ebv_included |  |
| NGC 4374 | 27.6983 | 0.0222 | 0.0007 | 0.0070 | 0.0027 | 0.0012 | 0.0235 |  | auto_from_median_bkg_check_and_Imean_included | auto_from_fast_detdist_detsamp_rerun_included | auto_irsa_sigma_ebv_included |  |
| NGC 4406 | 27.4875 | 0.0175 | 0.0004 | 0.0072 | 0.0038 | 0.0003 | 0.0193 |  | auto_from_median_bkg_check_and_Imean_included | auto_from_fast_detdist_detsamp_rerun_included | auto_irsa_sigma_ebv_included |  |
| NGC 4621 | 27.3500 | 0.0205 | 0.0001 | 0.0076 | 0.0141 | 0.0009 | 0.0261 |  | auto_from_median_bkg_check_and_Imean_included | auto_from_fast_detdist_detsamp_rerun_included | auto_irsa_sigma_ebv_included | truncated_file_warning;sersic_fit_warning |
| NGC 1549 | 27.6705 | 0.0173 | 0.0033 | 0.0073 | 0.0065 | 0.0003 | 0.0201 |  | auto_from_median_bkg_check_and_Imean_included | auto_from_fast_detdist_detsamp_rerun_included | auto_irsa_sigma_ebv_included | truncated_file_warning |
| NGC 3379 | 26.6536 | 0.0228 | 0.0004 | 0.0076 | 0.0156 | 0.0004 | 0.0287 |  | auto_from_median_bkg_check_and_Imean_included | auto_from_fast_detdist_detsamp_rerun_included | auto_irsa_sigma_ebv_included | truncated_file_warning |
