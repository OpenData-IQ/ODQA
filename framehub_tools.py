# tools/framehub_tools.py
from __future__ import annotations
from typing import List, Dict, Optional
from langchain_core.tools import tool
from framehub import FrameHub


def build_tools(hub: FrameHub):
    #@tool
    #def load_csv_string(csv_text: str, source_hint: Optional[str] = None, auto_clean: bool = True) -> str:
    #    """Load a CSV provided as a STRING. Trims header/footer junk and normalizes columns when auto_clean=True. Returns frame_id."""
    #    return hub.load_csv_string(csv_text, source_hint, auto_clean)
    @tool
    def drop_frame(frame_id: str) -> str:
        """Drop a frame from memory."""
        return hub.drop(frame_id)

    @tool
    def columns(frame_id: str) -> List[dict]:
        """List columns and dtypes for a frame_id."""
        return hub.columns(frame_id)

    #@tool
    #def preview(frame_id: str, n: int = 20) -> List[dict]:
    #    """Return up to n rows as JSON records for a frame_id."""
    #    return hub.preview(frame_id, n)

    @tool
    def select_cols(frame_id: str, cols: List[str]) -> str:
        """Project a subset of columns. Returns NEW frame_id."""
        return hub.select(frame_id, cols)

    @tool
    def filter_rows(frame_id: str, conditions: List[dict], logical_op: str = "AND") -> str:
        """Filter rows. conditions=[{col, op, value}] with ops: ==,!=,>,>=,<,<=,in,not_in,contains,startswith,endswith. Returns NEW frame_id."""
        return hub.filter(frame_id, conditions, logical_op)

    @tool
    def aggregate(frame_id: str, groupby: List[str], metrics: List[dict]) -> str:
        """Group-by and aggregate. metrics like [{op:'sum', col:'revenue', alias:'total_revenue'}] or [{op:'count'}]. Returns NEW frame_id."""
        return hub.aggregate(frame_id, groupby, metrics)

    @tool
    def sort_rows(frame_id: str, by: List[dict], limit: int | None = None) -> str:
        """Sort rows. by=[{col, desc?}], optional limit. Returns NEW frame_id."""
        return hub.sort(frame_id, by, limit)

    @tool
    def join_frames(left_id: str, right_id: str, on: List[dict], how: str = "inner") -> str:
        """Join two frames. 'on' can be [{col:'key'}] or [{left:'lkey', right:'rkey'}]. how in ['inner','left','right','outer']. Returns NEW frame_id."""
        return hub.join(left_id, right_id, on, how)  # type: ignore

    @tool
    def to_csv_string(frame_id: str, include_header: bool = True, limit: int | None = None) -> str:
        """Serialize a frame to CSV string (no files). Optionally limit rows."""
        return hub.to_csv_string(frame_id, include_header, limit)

    # return the actual tool objects LangChain creates from the decorators
    #return [load_csv_string, drop_frame, columns, preview, select_cols, filter_rows, aggregate, sort_rows, join_frames, to_csv_string]
    #return [drop_frame, columns, preview, select_cols, filter_rows, aggregate, sort_rows, join_frames,
    #        to_csv_string]
    return [drop_frame, columns, select_cols, filter_rows, aggregate, sort_rows, join_frames,
            to_csv_string]