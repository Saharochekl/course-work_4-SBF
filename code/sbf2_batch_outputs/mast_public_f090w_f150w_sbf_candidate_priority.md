# Public JWST/NIRCam F090W+F150W candidates for extending the SBF analysis

This list is broader than the strict SBF-literature match. It starts from all public JWST/NIRCam i2d observations with both F090W and F150W, keeps named/nearby galaxy-like targets, removes already processed GO-3055 galaxies, and assigns a practical suitability class.

## A_minus: nominal early-type, but inspect field carefully

| target | program | class | F150W obsids | F090W obsids | note |
|---|---|---|---|---|---|
| VV-191 | 1176 | strict_early_type | jw01176-o341_t005_nircam_clear-f150w | jw01176-o341_t005_nircam_clear-f090w | MAST marks elliptical, but target is VV-191 pair; likely contaminated by spiral/interacting structure. |

## B: SBF-literature dwarf elliptical, but Local Group/resolved regime

| target | program | class | F150W obsids | F090W obsids | note |
|---|---|---|---|---|---|
| NGC-147 | 4783 | local_group_dwarf_elliptical_sbf_literature | jw04783-o001_t001_nircam_clear-f150w | jw04783-o001_t001_nircam_clear-f090w | Known SBF-literature Local Group dwarf elliptical/spheroidal; likely heavily resolved, so pipeline assumptions must be checked. |
| NGC-185 | 4783 | local_group_dwarf_elliptical_sbf_literature | jw04783-o002_t002_nircam_clear-f150w | jw04783-o002_t002_nircam_clear-f090w | Known SBF-literature Local Group dwarf elliptical/spheroidal; likely heavily resolved, so pipeline assumptions must be checked. |
| NGC-205 | 4783 | local_group_dwarf_elliptical_sbf_literature | jw04783-o003_t003_nircam_clear-f150w | jw04783-o003_t003_nircam_clear-f090w | Known SBF-literature Local Group dwarf elliptical/spheroidal; likely heavily resolved, so pipeline assumptions must be checked. |

## C: dwarf/resolved population test, not direct giant-elliptical SBF

| target | program | class | F150W obsids | F090W obsids | note |
|---|---|---|---|---|---|
| AQUARIUS-DIRR | 4783 | local_group_dwarf_or_dwarf_spheroidal | jw04783-o004_t004_nircam_clear-f150w | jw04783-o004_t004_nircam_clear-f090w | Good for method stress-test on resolved stellar populations, weak for same calibration sample. |
| CETUS-DSPH | 4783 | local_group_dwarf_or_dwarf_spheroidal | jw04783-o007_t007_nircam_clear-f150w | jw04783-o007_t007_nircam_clear-f090w | Good for method stress-test on resolved stellar populations, weak for same calibration sample. |
| DRACO-II | 1334 | local_group_dwarf_or_dwarf_spheroidal | jw01334-o003_t003_nircam_clear-f150w | jw01334-o003_t003_nircam_clear-f090w | Good for method stress-test on resolved stellar populations, weak for same calibration sample. |
| IC-1613 | 4783 | local_group_dwarf_or_dwarf_spheroidal | jw04783-o008_t008_nircam_clear-f150w | jw04783-o008_t008_nircam_clear-f090w | Good for method stress-test on resolved stellar populations, weak for same calibration sample. |
| LEO-A | 4783 | local_group_dwarf_or_dwarf_spheroidal | jw04783-o006_t006_nircam_clear-f150w | jw04783-o006_t006_nircam_clear-f090w | Good for method stress-test on resolved stellar populations, weak for same calibration sample. |
| PEGASUS-DIRR | 4783 | local_group_dwarf_or_dwarf_spheroidal | jw04783-o005_t005_nircam_clear-f150w | jw04783-o005_t005_nircam_clear-f090w | Good for method stress-test on resolved stellar populations, weak for same calibration sample. |
| PEGASUS-W | 7119 | local_group_dwarf_or_dwarf_spheroidal | jw07119-o001_t001_nircam_clear-f150w | jw07119-o001_t001_nircam_clear-f090w | Good for method stress-test on resolved stellar populations, weak for same calibration sample. |
| TUCANAB | 4748 | local_group_dwarf_or_dwarf_spheroidal | jw04748-o001_t001_nircam_clear-f150w | jw04748-o001_t001_nircam_clear-f090w | Good for method stress-test on resolved stellar populations, weak for same calibration sample. |

## D: spiral/distance-ladder anchor; possible halo/bulge stress-test only

| target | program | class | F150W obsids | F090W obsids | note |
|---|---|---|---|---|---|
| LEO-P | 1617 | spiral_or_distance_ladder_anchor | jw01617-o001_t001_nircam_clear-f150w | jw01617-o001_t001_nircam_clear-f090w | Useful only if a smooth bulge/halo region can be isolated; disk, dust and Cepheids/SF fields may break current model. |
| M-81 | 1638 | spiral_or_distance_ladder_anchor | jw01638-o003_t003_nircam_clear-f150w | jw01638-o003_t003_nircam_clear-f090w | Useful only if a smooth bulge/halo region can be isolated; disk, dust and Cepheids/SF fields may break current model. |
| NGC-1448 | 1685 | spiral_or_distance_ladder_anchor | jw01685-o013_t008_nircam_clear-f150w;jw01685-o014_t008_nircam_clear-f150w | jw01685-o013_t008_nircam_clear-f090w;jw01685-o014_t008_nircam_clear-f090w | Useful only if a smooth bulge/halo region can be isolated; disk, dust and Cepheids/SF fields may break current model. |
| NGC-1559 | 1685 | spiral_or_distance_ladder_anchor | jw01685-o001_t001_nircam_clear-f150w;jw01685-o002_t001_nircam_clear-f150w | jw01685-o001_t001_nircam_clear-f090w;jw01685-o002_t001_nircam_clear-f090w | Useful only if a smooth bulge/halo region can be isolated; disk, dust and Cepheids/SF fields may break current model. |
| NGC-2403 | 1638 | spiral_or_distance_ladder_anchor | jw01638-o001_t001_nircam_clear-f150w | jw01638-o001_t001_nircam_clear-f090w | Useful only if a smooth bulge/halo region can be isolated; disk, dust and Cepheids/SF fields may break current model. |
| NGC-2525 | 2875 | spiral_or_distance_ladder_anchor | jw02875-o001_t001_nircam_clear-f150w;jw02875-o013_t001_nircam_clear-f150w | jw02875-o001_t001_nircam_clear-f090w;jw02875-o013_t001_nircam_clear-f090w | Useful only if a smooth bulge/halo region can be isolated; disk, dust and Cepheids/SF fields may break current model. |
| NGC-253 | 1638 | spiral_or_distance_ladder_anchor | jw01638-o004_t004_nircam_clear-f150w | jw01638-o004_t004_nircam_clear-f090w | Useful only if a smooth bulge/halo region can be isolated; disk, dust and Cepheids/SF fields may break current model. |
| NGC-300 | 1638 | spiral_or_distance_ladder_anchor | jw01638-o002_t002_nircam_clear-f150w | jw01638-o002_t002_nircam_clear-f090w | Useful only if a smooth bulge/halo region can be isolated; disk, dust and Cepheids/SF fields may break current model. |
| NGC-3147 | 2875 | spiral_or_distance_ladder_anchor | jw02875-o003_t003_nircam_clear-f150w | jw02875-o003_t003_nircam_clear-f090w | Useful only if a smooth bulge/halo region can be isolated; disk, dust and Cepheids/SF fields may break current model. |
| NGC-3370 | 2875 | spiral_or_distance_ladder_anchor | jw02875-o002_t007_nircam_clear-f150w | jw02875-o002_t007_nircam_clear-f090w | Useful only if a smooth bulge/halo region can be isolated; disk, dust and Cepheids/SF fields may break current model. |
| NGC-3447 | 2875 | spiral_or_distance_ladder_anchor | jw02875-o012_t009_nircam_clear-f150w | jw02875-o012_t009_nircam_clear-f090w | Useful only if a smooth bulge/halo region can be isolated; disk, dust and Cepheids/SF fields may break current model. |
| NGC-4258 | 1685 | spiral_or_distance_ladder_anchor | jw01685-o005_t003_nircam_clear-f150w;jw01685-o006_t003_nircam_clear-f150w | jw01685-o005_t003_nircam_clear-f090w;jw01685-o006_t003_nircam_clear-f090w | Useful only if a smooth bulge/halo region can be isolated; disk, dust and Cepheids/SF fields may break current model. |
| NGC-4258 | 2875 | spiral_or_distance_ladder_anchor | jw02875-o010_t008_nircam_clear-f150w | jw02875-o010_t008_nircam_clear-f090w | Useful only if a smooth bulge/halo region can be isolated; disk, dust and Cepheids/SF fields may break current model. |
| NGC-5468 | 1685 | spiral_or_distance_ladder_anchor | jw01685-o007_t004_nircam_clear-f150w;jw01685-o008_t004_nircam_clear-f150w | jw01685-o007_t004_nircam_clear-f090w;jw01685-o008_t004_nircam_clear-f090w | Useful only if a smooth bulge/halo region can be isolated; disk, dust and Cepheids/SF fields may break current model. |
| NGC-5584 | 1685 | spiral_or_distance_ladder_anchor | jw01685-o009_t005_nircam_clear-f150w;jw01685-o010_t005_nircam_clear-f150w | jw01685-o009_t005_nircam_clear-f090w;jw01685-o010_t005_nircam_clear-f090w | Useful only if a smooth bulge/halo region can be isolated; disk, dust and Cepheids/SF fields may break current model. |
| NGC-5643 | 1685 | spiral_or_distance_ladder_anchor | jw01685-o011_t006_nircam_clear-f150w;jw01685-o012_t006_nircam_clear-f150w | jw01685-o011_t006_nircam_clear-f090w;jw01685-o012_t006_nircam_clear-f090w | Useful only if a smooth bulge/halo region can be isolated; disk, dust and Cepheids/SF fields may break current model. |
| NGC-5861 | 2875 | spiral_or_distance_ladder_anchor | jw02875-o014_t006_nircam_clear-f150w | jw02875-o014_t006_nircam_clear-f090w | Useful only if a smooth bulge/halo region can be isolated; disk, dust and Cepheids/SF fields may break current model. |
| WFC3-ERS-FIELD | 1176 | spiral_or_distance_ladder_anchor | jw01176-o131_t016_nircam_clear-f150w | jw01176-o131_t016_nircam_clear-f090w | Useful only if a smooth bulge/halo region can be isolated; disk, dust and Cepheids/SF fields may break current model. |

