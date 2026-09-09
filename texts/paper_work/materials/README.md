# Article assets

`figures/` contains the exact images referenced by the English and Russian TeX
documents, including the commented annular-comparison figures. All TeX figure
paths are relative to `texts/paper_work/`. The Russian document is an older
scientific draft; collecting its images does not update its results.

`figure_sources.json` maps each publication copy to its producer output under
`runs/`. Those originals are intentionally retained: the analysis notebooks
still use them and can redraw them without rerunning the galaxy measurement.
Only the selected figures are copied here; unrelated diagnostics stay in runs.

From `code/`, after drawing figures manually:

```bash
py publish_article_assets.py
py publish_article_assets.py --check
```

The first command copies existing image bytes; it does not recalculate or redraw
anything. The second is read-only: it checks every publication image and checks
byte equality with any producer image still available. It can also check a
paper-only checkout, where the scientific producer outputs are not included.
The dedicated article figure builders refresh the relevant publication copy
immediately after saving a selected figure.

When adding a new figure to either TeX document, add its source and destination
to the manifest. Do not hand-edit the publication copy, as the next export
would replace it with the producer output.
