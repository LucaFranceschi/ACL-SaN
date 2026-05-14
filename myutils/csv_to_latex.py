r"""
build_table.py
--------------
Convert a multi-header pandas CSV to a LaTeX table where:
  - Rows are grouped by dataset (level-0 of the row MultiIndex), rendered
    as a \multirow label spanning all models for that dataset.
  - Best/second-best highlighting is computed *per dataset* (comparing
    models within each dataset for each metric column).
  - A visual separator (\midrule) is inserted between dataset groups.
  - The row index shows model (level-1) and epoch (level-2) as two
    separate columns after the dataset multirow cell.

Usage
-----
    python build_table.py results.csv \
        --header 0 1 2 \
        --index-cols 0 1 2 \
        --decimals 4
"""

import os
import argparse
import pandas as pd
from itertools import groupby
from pathlib import Path
import shutil

COLUMN_BEST_CRITERIA = {
    "cIoU_hat": "higher",
    "AUC":      "higher",
    "pIA_hat":  "lower",
    "AUC_N":    "higher",
}


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def fmt(val, decimals):
    if pd.isna(val):
        return "--"
    return f"{float(val):.{decimals}f}"

def fmt_snr(val):
    return r"$\infty$" if str(val) == "-1" else str(val)

def wrap(val_str, rank):
    if rank == "best":   return rf"\textbf{{{val_str}}}"
    if rank == "second": return rf"\textit{{{val_str}}}"
    return val_str


# ---------------------------------------------------------------------------
# Per-dataset ranking
# ---------------------------------------------------------------------------

def rank_within_dataset(series, decimals, higher=True):
    """Return {row_label -> 'best' | 'second'} for one (dataset, metric) slice.

    *series* is a pd.Series whose index values are the model/epoch labels
    (whatever we use to identify rows within a dataset group).
    """
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return {}
    unique_sorted = sorted(numeric.unique(), reverse=higher)
    best_val   = fmt(unique_sorted[0], decimals)
    second_val = fmt(unique_sorted[1], decimals) if len(unique_sorted) > 1 else None
    result = {}
    for idx, v in numeric.items():
        v = fmt(v, decimals)
        if v == best_val:
            result[idx] = "best"
        elif second_val is not None and v == second_val:
            result[idx] = "second"
    return result


def compute_highlights(df, criteria, decimals):
    """Return nested dict: highlights[dataset][col][model_key] -> rank str.

    *model_key* is the tuple (model, epoch) — the last two index levels.
    """
    highlights = {}
    # Group by dataset (level 0 of MultiIndex)
    for dataset, grp in df.groupby(level=0):
        highlights[dataset] = {}
        # Re-index by (model, epoch) so rank_within_dataset gets clean keys
        grp_reindexed = grp.copy()
        grp_reindexed.index = grp_reindexed.index.droplevel(0)  # drop dataset level
        for col in df.columns:
            metric = col[-1] if isinstance(col, tuple) else col
            higher = criteria.get(metric, "higher") == "higher"
            highlights[dataset][col] = rank_within_dataset(
                grp_reindexed[col], decimals, higher=higher,
            )
    return highlights


# ---------------------------------------------------------------------------
# Header helpers
# ---------------------------------------------------------------------------

def multicolumn_spans(keys):
    return [(k, len(list(grp))) for k, grp in groupby(keys)]


def render_header_row(spans, leading_cells):
    """Render one header row.

    *leading_cells* is a list of strings for the leftmost index columns
    (e.g. ['Dataset', 'Model', 'Epoch']).  An empty string means blank.
    """
    parts = list(leading_cells)
    for label, n in spans:
        if n == 1:
            parts.append(str(label))
        else:
            parts.append(rf"\multicolumn{{{n}}}{{c}}{{{label}}}")
    return " & ".join(parts) + r" \\"


def cmidrule_row(spans, start_col):
    rules = []
    col = start_col
    for _, n in spans:
        rules.append(rf"\cmidrule(lr){{{col}-{col + n - 1}}}")
        col += n
    return "    " + " ".join(rules)


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

MET_PRETTY = {
    "cIoU_hat": r"$\overline{cIoU}$",
    "pIA_hat":  r"$\overline{pIA}$",
    "AUC_N":    r"AUC$_N$",
}

# Number of leading index columns shown in the table:
#   Dataset (multirow)  |  Model  |  Epoch
N_INDEX_COLS = 3


def build_latex(df, decimals, label, criteria):
    cols     = list(df.columns)
    n_levels = df.columns.nlevels

    highlights = compute_highlights(df, criteria, decimals)

    # ---- column-format string ----
    col_fmt = "l" * (N_INDEX_COLS + 1) + "r" * len(cols)

    # ---- header span info ----
    at_keys  = [c[0] for c in cols]
    at_spans = multicolumn_spans(at_keys)

    if n_levels >= 2:
        l0l1_keys = [(c[0], c[1]) for c in cols]
        snr_spans = [(fmt_snr(k[1]), len(list(g))) for k, g in groupby(l0l1_keys)]

    if n_levels >= 3:
        met_keys = [c[2] for c in cols]

    # start_col for cmidrule: after the N_INDEX_COLS leading columns
    data_start = N_INDEX_COLS + 2
    table_name = label.replace('_', r"\_")

    lines = []
    lines.append(r"\centering")
    lines.append(
        rf"\caption{{Results for {table_name}. \textbf{{Bold}}: best per dataset; "
        r"\textit{italic}: second best per dataset. `--' means not applicable.}"
    )
    lines.append(rf"\label{{tab:{label}}}")
    lines.append(r"\setlength{\tabcolsep}{3pt}")
    lines.append(r"\renewcommand{\arraystretch}{1.05}")
    lines.append(rf"\begin{{tabular}}{{{col_fmt}}}")
    lines.append(r"  \toprule")

    # Header row 1
    lines.append(
        "  " + render_header_row(at_spans, [r"\textit{Audio type}", "Dataset", "Model", "Epoch"])
    )
    lines.append(cmidrule_row(at_spans, start_col=data_start))  # ← no +1

    # Header row 2
    if n_levels >= 2:
        lines.append(
            "  " + render_header_row(snr_spans, [r"\textit{SNR (dB)}", "", "", ""])
    )
        lines.append(cmidrule_row(snr_spans, start_col=data_start))  # ← no +1

    # Header row 3: Metric
    if n_levels >= 3:
        met_row = (
            " & ".join([r"\textit{Metric}", "", "", ""] + [MET_PRETTY.get(m, m) for m in met_keys])
            + r" \\"
        )
        lines.append("  " + met_row)

    lines.append(r"  \midrule")

    # ---- data rows, grouped by dataset ----
    datasets = df.index.get_level_values(0).unique()

    for d_idx, dataset in enumerate(datasets):
        grp = df.xs(dataset, level=0)          # index is now (model, epoch)
        n_rows = len(grp)
        dataset_tex = str(dataset).replace("_", r"\_")

        for r_idx, (model_epoch, row) in enumerate(grp.iterrows()):
            model, epoch = model_epoch          # unpack (model, epoch)
            model_tex = str(model).replace("_", r"\_")
            epoch_tex = str(epoch)

            # Dataset cell: multirow on first row of the group, blank otherwise
            if r_idx == 0:
                dataset_cell = rf"\multirow{{{n_rows}}}{{*}}{{{dataset_tex}}}"
            else:
                dataset_cell = ""

            # Build metric cells with per-dataset highlights
            ds_hl = highlights[dataset]
            cells = []
            for col in cols:
                val  = row[col]
                rank = ds_hl[col].get(model_epoch)
                cells.append(wrap(fmt(val, decimals), rank))

            row_str = (
                f"  & {dataset_cell} & {model_tex} & {epoch_tex} & "
                #  ^^ blank for the label column
                + " & ".join(cells)
                + r" \\"
            )
            lines.append(row_str)

        # Separator after each dataset group (except the last)
        if d_idx < len(datasets) - 1:
            lines.append(r"  \midrule")

    lines.append(r"  \bottomrule")
    lines.append(r"\end{tabular}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert a pandas CSV to a LaTeX table with per-dataset "
                    "best/second-best highlighting and dataset multirow grouping."
    )
    parser.add_argument("csv", type=Path, metavar="CSV")
    parser.add_argument("--cp_path", required=False, default=None, type=Path)
    parser.add_argument(
        "--header", nargs="+", type=int, default=[0, 1, 2], metavar="ROW",
        help="Row number(s) to use as column headers (default: 0 1 2).",
    )
    parser.add_argument(
        "--index-cols", nargs="+", type=int, default=[0, 1, 2], metavar="COL",
        help="Column number(s) to use as the row index (default: 0 1 2).",
    )
    parser.add_argument(
        "--decimals", type=int, default=4, metavar="N",
        help="Number of decimal places (default: 4).",
    )
    args = parser.parse_args()

    header     = args.header     if len(args.header)     > 1 else args.header[0]
    index_cols = args.index_cols if len(args.index_cols) > 1 else args.index_cols[0]

    df    = pd.read_csv(args.csv, header=header, index_col=index_cols)
    latex = build_latex(df, args.decimals, args.csv.stem, COLUMN_BEST_CRITERIA)

    out_path = os.path.splitext(os.path.abspath(args.csv))[0] + ".tex"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(latex + "\n")
    print(f"✓ Written to {out_path}")

    if args.cp_path:
        shutil.copy(out_path, args.cp_path)