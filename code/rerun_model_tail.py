import os
from pathlib import Path

os.environ["MPLCONFIGDIR"] = "/private/tmp/mplconfig_codex"
os.environ["XDG_CACHE_HOME"] = "/private/tmp/xdgcache_codex"
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

import base64
import builtins
import contextlib
import io
import time
import traceback

import matplotlib
import matplotlib.pyplot as plt
import nbformat
from nbformat.v4 import new_output

matplotlib.use("Agg")


SELECTED_CELLS = [
    2,
    4,
    6,
    8,
    11,
    13,
    15,
    17,
    19,
    20,
    22,
    24,
    26,
    28,
    30,
    32,
    33,
    34,
    36,
    38,
    40,
    42,
    46,
    48,
    50,
    51,
    53,
    55,
    62,
    64,
    66,
    68,
    70,
    74,
    76,
    78,
    80,
    81,
    82,
    85,
    87,
]


class StreamCapture(io.TextIOBase):
    def __init__(self, outputs):
        self.outputs = outputs

    def write(self, s):
        if not s:
            return 0
        if self.outputs and self.outputs[-1].output_type == "stream" and self.outputs[-1].name == "stdout":
            self.outputs[-1].text += s
        else:
            self.outputs.append(new_output("stream", name="stdout", text=s))
        return len(s)

    def flush(self):
        return None


def make_display(outputs):
    def display(obj):
        bundle = {"text/plain": repr(obj)}
        if hasattr(obj, "_repr_html_"):
            try:
                html = obj._repr_html_()
            except Exception:
                html = None
            if html:
                bundle["text/html"] = html
        outputs.append(new_output("display_data", data=bundle, metadata={}))

    return display


def make_show(outputs):
    def show(*args, **kwargs):
        for num in list(plt.get_fignums()):
            fig = plt.figure(num)
            buf = io.BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight")
            png_b64 = base64.b64encode(buf.getvalue()).decode("ascii")
            outputs.append(
                new_output(
                    "display_data",
                    data={"image/png": png_b64, "text/plain": repr(fig)},
                    metadata={},
                )
            )
            plt.close(fig)

    return show


def main():
    nb_path = Path("sbf-1.ipynb")
    nb = nbformat.read(nb_path, as_version=4)

    ns = {"__builtins__": __builtins__}
    start_all = time.time()

    for idx in SELECTED_CELLS:
        cell = nb.cells[idx]
        src = "".join(cell.source)
        outputs = []
        ns["display"] = make_display(outputs)
        old_show = plt.show
        plt.show = make_show(outputs)
        t0 = time.time()
        try:
            with contextlib.redirect_stdout(StreamCapture(outputs)):
                exec(src, ns)
        except Exception as exc:
            tb = traceback.format_exc().splitlines()
            outputs.append(new_output("error", ename=type(exc).__name__, evalue=str(exc), traceback=tb))
            cell.outputs = outputs
            cell.execution_count = idx
            nbformat.write(nb, nb_path)
            print(f"FAILED cell {idx}: {type(exc).__name__}: {exc}")
            raise
        finally:
            plt.show = old_show

        cell.outputs = outputs
        cell.execution_count = idx
        nbformat.write(nb, nb_path)
        print(f"EXECUTED cell {idx} in {time.time() - t0:.1f}s")

    print(f"ALL_DONE {len(SELECTED_CELLS)} cells in {time.time() - start_all:.1f}s")


if __name__ == "__main__":
    _orig_print = builtins.print

    def print(*args, **kwargs):
        _orig_print(f"[driver {time.strftime('%H:%M:%S')}]", *args, **kwargs)

    main()
