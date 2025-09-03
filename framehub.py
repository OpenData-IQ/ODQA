from __future__ import annotations
import csv, io, re, uuid
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Literal
import polars as pl

# ---------------- CSV cleaning from *string* ----------------

COMMON_NULLS = {"", "na", "n/a", "null", "none", "-", "--", "nan"}

def _detect_dialect(sample: str) -> csv.Dialect:
    try:
        return csv.Sniffer().sniff(sample, delimiters=[",",";","|","\t"])
    except Exception:
        class Simple(csv.Dialect):
            delimiter=","; quotechar='"'; doublequote=True
            skipinitialspace=True; lineterminator="\n"; quoting=csv.QUOTE_MINIMAL
        return Simple()

def _normalize_header(header: List[str]) -> List[str]:
    out, seen = [], set()
    for h in header:
        n = re.sub(r"[^0-9a-zA-Z_ ]","", (h or "").strip()).lower().replace(" ","_") or "col"
        base, k = n, 1
        while n in seen:
            k += 1; n = f"{base}_{k}"
        seen.add(n); out.append(n)
    return out

def clean_csv_text_to_polars(text: str, extra_nulls: Optional[List[str]] = None) -> pl.DataFrame:
    """
    Trim non-tabular lines at top/bottom, normalize headers, collapse common null tokens.
    Input is a *CSV string*, not a file.
    """
    # normalize newlines
    if "\r\n" in text or "\r" in text:
        text = text.replace("\r\n","\n").replace("\r","\n")

    sample = text[:100_000]
    dialect = _detect_dialect(sample)

    lines = text.splitlines(True)
    # sample top for width + header guess
    sample_reader = csv.reader(io.StringIO("".join(lines[:2000])), dialect=dialect)
    sample_rows = list(sample_reader) or [[]]

    width_counts = Counter(len(r) for r in sample_rows if r)
    ncols = max(width_counts, key=width_counts.get) if width_counts else 0
    if ncols <= 1:
        ncols = max((len(r) for r in sample_rows), default=2)

    # find first width-consistent row = header_idx
    header_idx = 0
    for i, r in enumerate(sample_rows):
        if len(r) == ncols:
            header_idx = i; break

    # find last width-consistent row from bottom
    last_idx = len(lines) - 1
    for i in range(len(lines)-1, header_idx, -1):
        r = next(csv.reader([lines[i]], dialect=dialect), None)
        if r and len(r) == ncols:
            last_idx = i; break

    core = lines[header_idx:last_idx+1]
    data = list(csv.reader(io.StringIO("".join(core)), dialect=dialect))
    if not data:
        return pl.DataFrame()

    header = _normalize_header(data[0])
    body = [r for r in data[1:] if len(r) == len(header)]

    nulls = {*(extra_nulls or []), *COMMON_NULLS}
    nulls = {s.lower() for s in nulls}

    cleaned_rows: List[List[Optional[str]]] = []
    for row in body:
        cleaned = []
        for v in row:
            s = re.sub(r"\s+"," ", (v or "")).strip()
            cleaned.append(None if s.lower() in nulls else s)
        cleaned_rows.append(cleaned)

    return pl.DataFrame(cleaned_rows, schema=header)

# ---------------- FrameHub (multi-frame, functional ops) ----------------

@dataclass
class FrameInfo:
    source_hint: Optional[str] = None  # e.g., "sales_2024.csv" or "upload#123"
    steps: List[str] = None

class FrameHub:
    """
    Manage multiple Polars DataFrames in memory (keyed by frame_id).
    All transforms are *functional*: they return a NEW frame_id.
    """
    def __init__(self):
        self.frames: Dict[str, pl.DataFrame] = {}
        self.meta: Dict[str, FrameInfo] = {}

    def __getitem__(self, frame_id: str):
        return self.frames[frame_id]

    def _new_id(self) -> str:
        return uuid.uuid4().hex[:8]

    # ---- Load / Register ----
    def load_csv_string(self, csv_text: str, source_hint: Optional[str]=None, auto_clean: bool=True) -> str:
        df = clean_csv_text_to_polars(csv_text) if auto_clean else pl.read_csv(io.BytesIO(csv_text.encode("utf-8")))
        # normalize headers again (idempotent)
        df = df.rename({c: c.strip().lower().replace(" ","_") for c in df.columns})
        fid = self._new_id()
        self.frames[fid] = df
        self.meta[fid] = FrameInfo(source_hint=source_hint, steps=["load_csv_string(auto_clean=%s)" % auto_clean])
        return fid

    def register_frame(self, df: pl.DataFrame, source_hint: Optional[str]=None) -> str:
        fid = self._new_id()
        self.frames[fid] = df
        self.meta[fid] = FrameInfo(source_hint=source_hint, steps=["register"])
        return fid

    def drop(self, frame_id: str) -> Literal["ok"]:
        self.frames.pop(frame_id, None)
        self.meta.pop(frame_id, None)
        return "ok"

    # ---- Introspection ----
    def columns(self, frame_id: str) -> List[Dict[str,str]]:
        df = self.frames[frame_id]
        return [{"name": n, "dtype": str(t)} for n, t in zip(df.columns, df.dtypes)]

    def preview(self, frame_id: str, n: int = 20) -> List[Dict[str, Any]]:
        df = self.frames[frame_id]
        return df.head(n).to_dicts()

    # ---- Transforms (return NEW frame_id) ----
    def select(self, frame_id: str, cols: List[str]) -> str:
        df = self.frames[frame_id].select([pl.col(c) for c in cols])
        out = self.register_frame(df, source_hint=self.meta[frame_id].source_hint)
        self.meta[out].steps = [*self.meta[frame_id].steps, f"select({cols})"]
        return out

    def filter(self, frame_id: str, conditions: List[Dict[str, Any]], logical_op: Literal["AND","OR"]="AND") -> str:
        df = self.frames[frame_id]
        exprs = []
        for c in conditions:
            col, op, val = c["col"], c["op"], c.get("value")
            s = pl.col(col)
            match op:
                case "==": e = s == val
                case "!=": e = s != val
                case ">":  e = s >  val
                case ">=": e = s >= val
                case "<":  e = s <  val
                case "<=": e = s <= val
                case "in": e = s.is_in(val if isinstance(val, list) else [val])
                case "not_in": e = ~s.is_in(val if isinstance(val, list) else [val])
                case "contains": e = s.cast(pl.Utf8).str.contains(val, literal=True)
                case "startswith": e = s.cast(pl.Utf8).str.starts_with(val)
                case "endswith": e = s.cast(pl.Utf8).str.ends_with(val)
                case _: raise ValueError(f"Unsupported op: {op}")
            exprs.append(e)
        mask = exprs[0]
        for e in exprs[1:]:
            mask = mask & e if logical_op == "AND" else mask | e
        out_df = df.filter(mask)
        out = self.register_frame(out_df, source_hint=self.meta[frame_id].source_hint)
        self.meta[out].steps = [*self.meta[frame_id].steps, f"filter({conditions}, {logical_op})"]
        return out

    def aggregate(self, frame_id: str, groupby: List[str], metrics: List[Dict[str,str]]) -> str:
        df = self.frames[frame_id]
        aggs = []
        for m in metrics:
            op, col = m["op"].lower(), m.get("col")
            alias = m.get("alias") or (f"{op}_{col}" if col else op)
            match op:
                case "count":  e = pl.count().alias(alias)
                case "sum":    e = pl.col(col).sum().alias(alias)
                case "avg"|"mean": e = pl.col(col).mean().alias(alias)
                case "min":    e = pl.col(col).min().alias(alias)
                case "max":    e = pl.col(col).max().alias(alias)
                case "median": e = pl.col(col).median().alias(alias)
                case "nunique":e = pl.col(col).n_unique().alias(alias)
                case _: raise ValueError(f"Unsupported agg: {op}")
            aggs.append(e)
        out_df = df.group_by(groupby).agg(aggs)
        out = self.register_frame(out_df, source_hint=self.meta[frame_id].source_hint)
        self.meta[out].steps = [*self.meta[frame_id].steps, f"aggregate({groupby},{metrics})"]
        return out

    def sort(self, frame_id: str, by: List[Dict[str, Any]], limit: Optional[int]=None) -> str:
        df = self.frames[frame_id]
        cols = [b["col"] for b in by]
        desc = [bool(b.get("desc", False)) for b in by]
        out_df = df.sort(cols, descending=desc)
        if limit: out_df = out_df.head(limit)
        out = self.register_frame(out_df, source_hint=self.meta[frame_id].source_hint)
        self.meta[out].steps = [*self.meta[frame_id].steps, f"sort({by}, limit={limit})"]
        return out

    def join(self, left_id: str, right_id: str, on: List[Dict[str,str]], how: Literal["inner","left","right","outer"]="inner") -> str:
        left = self.frames[left_id]; right = self.frames[right_id]
        # on format: [{"left":"l_key","right":"r_key"}] or [{"col":"key"}] for same-name
        left_on = [o.get("left", o.get("col")) for o in on]
        right_on = [o.get("right", o.get("col")) for o in on]
        out_df = left.join(right, left_on=left_on, right_on=right_on, how=how)
        out = self.register_frame(out_df, source_hint=f"join({left_id},{right_id})")
        self.meta[out].steps = ["join"]  # keep concise
        return out

    def to_csv_string(self, frame_id: str, include_header: bool = True, limit: Optional[int]=None) -> str:
        df = self.frames[frame_id]
        if limit: df = df.head(limit)
        return df.write_csv(include_header=include_header)
