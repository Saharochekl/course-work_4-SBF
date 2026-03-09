#!/usr/bin/env python3
# Extract SCI extensions from JWST multi-extension FITS without resampling/scaling
# Robust against junk files, gzip, noisy headers, and Python 3.13 warning logger quirks.
# Outputs one PrimaryHDU per SCI with original WCS header preserved.

from pathlib import Path
import os
import warnings
import gzip
import io
import numpy as np

# --- kill warnings/log spam BEFORE importing astropy ---
os.environ.setdefault("PYTHONWARNINGS", "ignore")
os.environ.setdefault("ASTROPY_LOG_LEVEL", "ERROR")
warnings.filterwarnings("ignore")
warnings.showwarning = lambda *a, **k: None

from astropy.io import fits
from astropy.utils.exceptions import AstropyWarning, AstropyUserWarning
from astropy import log as astropy_log

# hard-mute astropy/logger warning hook and Python warnings to avoid pathlib crashes on 3.13
try:
    import astropy.logger as _astropy_logger
    _astropy_logger._showwarning = lambda *a, **k: None  # kill astropy's custom hook
except Exception:
    pass
warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")
# kill both classic and new warning pathways
warnings._showwarning = lambda *a, **k: None
try:
    warnings._showwarnmsg = lambda msg: None
except Exception:
    pass

# Final safety net for astropy warnings
warnings.filterwarnings("ignore", category=AstropyWarning)
warnings.filterwarnings("ignore", category=AstropyUserWarning)
try:
    astropy_log.setLevel('ERROR')
except Exception:
    pass

# --- I/O roots (relative to this script) ---
_SCRIPT_DIR = Path(__file__).resolve().parent
IN  = (_SCRIPT_DIR / "../data/nrcblong").resolve()          # where original FITS live
OUT = (_SCRIPT_DIR / "../data/sci_only_nrcblong").resolve() # where to write SCI-only FITS
OUT.mkdir(parents=True, exist_ok=True)

_MIN_FITS_SIZE = 2880  # one FITS block
_MAX_SCAN_BLOCKS = 16  # up to 16*2880 = 46,080 bytes from start
_ASCII_OK = set(range(9, 127)) | {32}  # tab..~ plus space


def _read_head_bytes(path: Path, nbytes: int) -> bytes:
    """Read up to nbytes from start; transparently handle gzip; return bytes or b'' on error."""
    try:
        with open(path, 'rb') as fh:
            magic = fh.read(2)
        if magic == b"\x1f\x8b":
            with gzip.open(path, 'rb') as g:
                return g.read(nbytes)
        with open(path, 'rb') as fh:
            return fh.read(nbytes)
    except Exception:
        return b''


def quick_fits_header_ok(path: Path) -> bool:
    """Very fast sanity-check: SIMPLE at start, ASCII-like header, END within first blocks."""
    try:
        if path.name.startswith('._'):
            return False
        if path.stat().st_size < _MIN_FITS_SIZE:
            return False
    except Exception:
        return False

    head = _read_head_bytes(path, 2880 * _MAX_SCAN_BLOCKS)
    if len(head) < 80:
        return False
    if not head.startswith(b'SIMPLE'):
        # gzip may start with gzip header; we already handled it in _read_head_bytes
        return False
    # header must be a sequence of 80-char cards until END card
    # check ASCII-ish bytes and presence of END at card boundary
    # speed: scan by 80
    found_end = False
    limit = (len(head) // 80) * 80
    for off in range(0, min(limit, 2880 * _MAX_SCAN_BLOCKS), 80):
        card = head[off:off+80]
        if any((c not in _ASCII_OK) for c in card):
            # non-ascii in header region -> bail
            return False
        if card.startswith(b'END'):
            found_end = True
            break
    return found_end


def open_fits_safe(path: Path):
    """Open FITS safely; reject obvious junk before letting astropy parse headers."""
    if not quick_fits_header_ok(path):
        raise ValueError('corrupt-or-notfits')
    # gzip handled transparently by fits if given a file object
    with open(path, 'rb') as fh:
        magic = fh.read(2)
    if magic == b"\x1f\x8b":
        gz = gzip.open(path, 'rb')
        try:
            return fits.open(gz, memmap=False, ignore_missing_end=True)
        except Exception:
            gz.close()
            raise
    return fits.open(path, memmap=False, ignore_missing_end=True)


def sci_hdus_lite(hdul, fname: str):
    """Try to locate SCI HDUs without iterating entire file. Prefer ext=1, then small probe 1..9."""
    scis = []
    # Strategy: probe first several extensions by index only; avoid full iteration
    for ext in (1, 2, 3, 4):
        try:
            h = hdul[ext]
        except Exception:
            break
        try:
            if getattr(h, 'name', '') == 'SCI':
                scis.append(h)
        except Exception as e:
            print(f"[skip-ext] {fname}: ext#{ext}: {e}")
            break
    # Some products may have SCI at ext 0 (rare). Check it last.
    try:
        h0 = hdul[0]
        if getattr(h0, 'name', '') == 'SCI':
            scis.append(h0)
    except Exception:
        pass
    return scis


def dump_exts_safe(hdul) -> str:
    out = []
    # only peek first 12 HDUs to avoid tripping on broken tails
    for i in range(0, 12):
        try:
            h = hdul[i]
        except Exception:
            break
        try:
            name = getattr(h, 'name', str(i))
            ver  = h.header.get('EXTVER', 0)
            shp  = getattr(h, 'data', None)
            shp  = tuple(shp.shape) if shp is not None else ()
            out.append(f"[{i}:{name}{ver if ver else ''}:{'x'.join(map(str,shp))}]")
        except Exception:
            out.append(f"[{i}:?]")
            break
    return ' '.join(out)


# --- main ---

total = wrote = skips = 0

for f in IN.rglob("*_i2d.fits"):
    total += 1
    try:
        hdul = open_fits_safe(f)
        with hdul:
            scis = sci_hdus_lite(hdul, f.name)
            if not scis:
                print(f"[skip-no-SCI] {f.name} :: {dump_exts_safe(hdul)}")
                skips += 1
                continue
            if len(scis) == 1:
                h = scis[0]
                data = np.asarray(h.data, dtype=np.float32)
                hdr  = h.header.copy()
                out = OUT / (f.stem + "_SCI.fits")
                fits.PrimaryHDU(data=data, header=hdr).writeto(out, overwrite=True)
                wrote += 1
            else:
                for h in scis:
                    ver = h.header.get('EXTVER', 1)
                    data = np.asarray(h.data, dtype=np.float32)
                    hdr  = h.header.copy()
                    out = OUT / (f.stem + f"_SCI{ver}.fits")
                    fits.PrimaryHDU(data=data, header=hdr).writeto(out, overwrite=True)
                    wrote += 1
    except ValueError as e:
        print(f"[skip-notfits] {f.name}: {e}")
        skips += 1
    except UnicodeDecodeError as e:
        print(f"[skip-corrupt-header] {f.name}: {e}")
        skips += 1
    except IndexError as e:
        print(f"[skip-bad-hdu] {f.name}: {e}")
        skips += 1
    except Exception as e:
        print(f"[skip] {f.name}: {e}")
        skips += 1

print(f"Done. scanned={total}, written={wrote}, skipped={skips}")