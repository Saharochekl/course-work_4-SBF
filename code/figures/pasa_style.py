"""PASA: только оформление существующих artists / Style existing artists only.

No arrays, uncertainties, fitted coefficients or sample membership are changed.
The final canvas matches the official two-column text width; include it at
``\\textwidth`` rather than reducing the text a second time in LaTeX.
"""

from textwrap import fill

from matplotlib.text import Text


# pas.cls uses TeX points (72.27 per inch), not PDF's 72-point unit.
WIDTH_INCH = 515 / 72.27
# Final-size typography: ticks/NGC labels 8 pt, axes 9 pt, headings 10 pt.
TICK_PT, LABEL_PT, TITLE_PT = 8, 9, 10


def british_spelling(text):
    """Change visible prose only; identifiers and numerical data stay intact."""
    for old, new in (("Color", "Colour"), ("color", "colour"),
                     ("normalization", "normalisation"),
                     ("Normalized", "Normalised"), ("normalized", "normalised"),
                     ("Winsorized", "Winsorised"), ("winsorized", "winsorised")):
        text = text.replace(old, new)
    return text


def prepare_figure(fig, stem):
    """Fit the original plotting artists to the journal page, without refitting."""
    panels = len(fig.axes)
    # Panel heights provide room for galaxy labels and 14-row budgets; they are
    # presentation choices, not changes to data ranges or histogram bins.
    height = {2: 3.8, 3: 3.2, 4: 5.8, 6: 7.8}.get(panels, 4.4)
    if "error_budget" in stem or "loo_residuals" in stem:
        height = 4.8
    elif "loo_recovery" in stem:
        height = 5.4
    elif "psf_size" in stem:
        height = 3.3
    fig.set_size_inches(WIDTH_INCH, height, forward=True)
    for text in fig.findobj(Text):
        text.set_text(british_spelling(text.get_text()))
        text.set_fontfamily("DejaVu Sans")
        text.set_fontsize(TICK_PT)

    for ax in fig.axes:
        ax.title.set_fontsize(TITLE_PT)
        title = ax.get_title()
        if "$" not in title and panels > 1:
            ax.set_title(fill(title, width=25 if panels == 3 else 38), fontsize=TITLE_PT)
        ax.xaxis.label.set_fontsize(LABEL_PT)
        ax.yaxis.label.set_fontsize(LABEL_PT)
        if panels == 3 and "Jensen et al." in ax.get_xlabel():
            ax.set_xlabel(ax.get_xlabel().replace(", Jensen", "\nJensen"), fontsize=LABEL_PT)
        ax.tick_params(which="major", labelsize=TICK_PT, length=4, width=0.8)
        ax.tick_params(which="minor", length=2, width=0.6)
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)
        for grid in ax.get_xgridlines() + ax.get_ygridlines():
            grid.set_linewidth(0.5)
        for line in ax.lines:
            line.set_linewidth(min(line.get_linewidth(), 1.3))
            line.set_markersize(min(line.get_markersize(), 4.8))
        for collection in ax.collections:
            if collection.__class__.__name__ == "LineCollection":
                collection.set_linewidth(0.8)
        for text in ax.texts:
            text.set_fontsize(LABEL_PT if panels == 1 else TICK_PT)

    if fig._suptitle is not None:
        fig._suptitle.set_fontsize(TITLE_PT)
    if "error_budget" in stem:
        # The nine-component legend must not cover the horizontal variance bars.
        ax = fig.axes[0]
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, [british_spelling(label) for label in labels],
                  loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3,
                  frameon=False, fontsize=TICK_PT, columnspacing=1.2,
                  handletextpad=0.5, borderaxespad=0)
    # Figure-level legends sit below multi-panel axes. Axis-level legends retain
    # their original placement and are included by tight_layout.
    bottom = 0.06 if fig.legends else 0
    top = 0.955 if fig._suptitle is not None else 1
    fig.tight_layout(pad=0.7, h_pad=0.9, w_pad=0.8, rect=(0, bottom, 1, top))
