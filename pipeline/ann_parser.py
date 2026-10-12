"""Step 1-2: .ann metadata parsing for UAVSAR SLC scenes."""
import re

_LINE_RE = re.compile(r"^\s*([^=;()]+?)\s*(\([^)]*\))?\s*=\s*([^;]+?)\s*(;.*)?$")


def parse_ann(path):
    """Parse a UAVSAR .ann file into {key: value} with numeric coercion."""
    out = {}
    with open(path, "r", errors="ignore") as f:
        for line in f:
            m = _LINE_RE.match(line)
            if not m:
                continue
            key, _, val, _ = m.groups()
            key = key.strip()
            val = val.strip()
            out[key] = _coerce(val)
    return out


def _coerce(val):
    try:
        return int(val)
    except ValueError:
        pass
    try:
        return float(val)
    except ValueError:
        return val


_ROWS_SUFFIXES = ("set_rows", "rows")
_COLS_SUFFIXES = {"set_rows": ("set_cols",), "rows": ("columns", "cols")}


def find_slc_segments(ann):
    """Return list of (rows, cols, key_prefix) for each range segment found.

    Key naming varies by data version: older exports use
    'slc_1_1x1 rows' / 'slc_1_1x1 columns' (segment 1, segment 2 would be
    'slc_1_2x8 rows' / ...); current UAVSAR .ann files use
    'slc_mag.set_rows' / 'slc_mag.set_cols' instead. Both are matched here
    and deduplicated by (rows, cols), since slc_mag/slc_phase/etc all
    describe the same underlying raw .slc binary dimensions.
    """
    seen = {}
    for key in ann:
        if not key.startswith("slc_"):
            continue
        for rsuf in _ROWS_SUFFIXES:
            if not key.endswith(rsuf):
                continue
            prefix = key[: -len(rsuf)]
            for csuf in _COLS_SUFFIXES[rsuf]:
                cols_key = f"{prefix}{csuf}"
                if cols_key in ann:
                    dims = (int(ann[key]), int(ann[cols_key]))
                    seen.setdefault(dims, prefix.strip(" ._") or "slc")
                    break
            break
    segments = [(rows, cols, label) for (rows, cols), label in seen.items()]
    segments.sort(key=lambda s: s[2])
    return segments


def nodata_value(ann, default=0.0):
    for key in ("slc_no_data", "SLC No Data", "no_data"):
        if key in ann:
            try:
                return float(ann[key])
            except (TypeError, ValueError):
                pass
    return default
