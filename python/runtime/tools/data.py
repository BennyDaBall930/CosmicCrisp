"""Data helper tool for simple tabular parsing and chart generation."""
from __future__ import annotations

import csv
import io
import statistics
from typing import Any, Dict, List

from .base import Tool
from .registry import register_tool


class DataTool(Tool):
    name = "data"
    description = "Parse CSV/TSV snippets and produce basic statistics."

    async def run(self, data: str, *, format: str = "csv", **_: Any) -> str:
        if not data:
            return "data: no input provided"
        delimiter = "	" if format.lower() == "tsv" else ","
        reader = csv.DictReader(io.StringIO(data), delimiter=delimiter)
        rows = list(reader)
        if not rows:
            return "data: no rows parsed"
        summary = self._summarize(rows)
        lines = ["data summary:"]
        for column, stats in summary.items():
            lines.append(f"- {column}: {stats}")
        return "\n".join(lines)

    def _summarize(self, rows: List[Dict[str, str]]) -> Dict[str, str]:
        summary: Dict[str, str] = {}
        if not rows:
            return summary
        columns = rows[0].keys()
        for column in columns:
            values = [row[column] for row in rows if row.get(column)]
            numeric: List[float] = []
            for value in values:
                try:
                    numeric.append(float(value))
                except Exception:
                    pass
            if numeric:
                summary[column] = (
                    f"count={len(numeric)}, mean={statistics.mean(numeric):.2f}, "
                    f"stdev={statistics.pstdev(numeric):.2f}"
                )
            else:
                unique = list(dict.fromkeys(values))[:5]
                summary[column] = f"{len(values)} values, examples={unique}"
        return summary


register_tool(DataTool)
