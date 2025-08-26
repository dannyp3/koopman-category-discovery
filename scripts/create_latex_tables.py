import pandas as pd
from pathlib import Path

# ----- CONFIG -----
# Option A: point to a CSV; Option B: leave as None if you'll pass df directly.
CSV_PATH = "../data/metrics.csv"  # e.g., "results_96.csv"
OUT_DIR = "../data/latex_tables"  # where to save the .tex files
PRECISION = 1  # decimals in LaTeX (e.g., 1 -> 00.0)

# Column names (combined Hungarian metrics)
K_STRICT = "kcm_combined_strict_hungarian"
K_GREEDY = "kcm_combined_greedy_hungarian"
B_STRICT = "basic_combined_strict_hungarian"
B_GREEDY = "basic_combined_greedy_hungarian"

# Experiment structure
HASHES = [4, 6, 8]
BLOCKS_1D = [("Full (10 cats)", 1, 10, [(7, 3), (3, 7)]),
             ("Small (4 cats)", 1, 4,  [(3, 1), (2, 2)])]
BLOCKS_3D = [("Full (7 cats)", 3, 7,  [(5, 2), (3, 4)]),
             ("Small (4 cats)", 3, 4,  [(3, 1), (2, 2)])]

def fmt(x):
    if pd.isna(x): 
        return "--"
    try:
        x = float(x)
        if 0.0 <= x <= 1.0:  # in case scores are in [0,1]
            x *= 100.0
        return f"{round(x):d}"
    except Exception:
        return "--"


def get_val(df, dims, cats, train, test, hash_sz, noisy, col):
    row = df[(df["dimensions"] == dims) &
             (df["num_cats"] == cats) &
             (df["train_classes"] == train) &
             (df["test_classes"] == test) &
             (df["output_dims"] == hash_sz) &
             (df["noisy_data"] == noisy)]
    if row.empty: return float("nan")
    return row.iloc[0][col]

def block_rows(df, dims, cats, splits, include_delta=False):
    rows = []
    for (tr, te) in splits:
        label = f"train {tr} / test {te}"
        for h in HASHES:
            # Clean/Noisy for each metric
            ks_c = get_val(df, dims, cats, tr, te, h, False, K_STRICT)
            ks_n = get_val(df, dims, cats, tr, te, h, True,  K_STRICT)
            kg_c = get_val(df, dims, cats, tr, te, h, False, K_GREEDY)
            kg_n = get_val(df, dims, cats, tr, te, h, True,  K_GREEDY)
            bs_c = get_val(df, dims, cats, tr, te, h, False, B_STRICT)
            bs_n = get_val(df, dims, cats, tr, te, h, True,  B_STRICT)
            bg_c = get_val(df, dims, cats, tr, te, h, False, B_GREEDY)
            bg_n = get_val(df, dims, cats, tr, te, h, True,  B_GREEDY)

            if include_delta:
                ks_d = (ks_n - ks_c) if pd.notna(ks_c) and pd.notna(ks_n) else float("nan")
                kg_d = (kg_n - kg_c) if pd.notna(kg_c) and pd.notna(kg_n) else float("nan")
                bs_d = (bs_n - bs_c) if pd.notna(bs_c) and pd.notna(bs_n) else float("nan")
                bg_d = (bg_n - bg_c) if pd.notna(bg_c) and pd.notna(bg_n) else float("nan")
                line = (f"{label} & {h} & "
                        f"{fmt(ks_c)} & {fmt(ks_n)} & {fmt(ks_d)} & "
                        f"{fmt(kg_c)} & {fmt(kg_n)} & {fmt(kg_d)} & "
                        f"{fmt(bs_c)} & {fmt(bs_n)} & {fmt(bs_d)} & "
                        f"{fmt(bg_c)} & {fmt(bg_n)} & {fmt(bg_d)} \\\\")
            else:
                line = (f"{label} & {h} & "
                        f"{fmt(ks_c)} & {fmt(ks_n)} & "
                        f"{fmt(kg_c)} & {fmt(kg_n)} & "
                        f"{fmt(bs_c)} & {fmt(bs_n)} & "
                        f"{fmt(bg_c)} & {fmt(bg_n)} \\\\")
            rows.append(line)
    return rows


import string  # <-- add this near the top


def table_common_header(caption, label, noisy_cols_with_delta=False):
    if noisy_cols_with_delta:
        cols = (r"\multicolumn{3}{c}{\textbf{Koopman Strict-H}} & "
                r"\multicolumn{3}{c}{\textbf{Koopman Greedy-H}} & "
                r"\multicolumn{3}{c}{\textbf{Basic Strict-H}} & "
                r"\multicolumn{3}{c}{\textbf{Basic Greedy-H}}")
        sub = (r"\textbf{Clean} & \textbf{Noisy} & {$\Delta$} & "
               r"\textbf{Clean} & \textbf{Noisy} & {$\Delta$} & "
               r"\textbf{Clean} & \textbf{Noisy} & {$\Delta$} & "
               r"\textbf{Clean} & \textbf{Noisy} & {$\Delta$}")
        colspec = r"l c S S S  S S S  S S S  S S S"
        cmid = (r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}"
                r"\cmidrule(lr){9-11}\cmidrule(lr){12-14}")
    else:
        cols = (r"\multicolumn{2}{c}{\textbf{Koopman Strict-H}} & "
                r"\multicolumn{2}{c}{\textbf{Koopman Greedy-H}} & "
                r"\multicolumn{2}{c}{\textbf{Basic Strict-H}} & "
                r"\multicolumn{2}{c}{\textbf{Basic Greedy-H}}")
        sub = (r"\textbf{Clean} & \textbf{Noisy} & "
               r"\textbf{Clean} & \textbf{Noisy} & "
               r"\textbf{Clean} & \textbf{Noisy} & "
               r"\textbf{Clean} & \textbf{Noisy}")
        colspec = r"l c S S  S S  S S  S S"
        cmid = (r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}"
                r"\cmidrule(lr){7-8}\cmidrule(lr){9-10}")

    tmpl = string.Template(r"""
\begin{table*}[t]
\centering
\caption{$caption}
\label{$label}
\sisetup{table-format=2.$prec,detect-weight=true,detect-inline-weight=math}
\renewcommand{\arraystretch}{1.12}
\setlength{\tabcolsep}{3.5pt}
\begin{tabularx}{\textwidth}{$colspec}
\toprule
& & $cols \\
$cmid
\textbf{Group / Split} & \textbf{Hash} & $sub \\
\midrule
""")
    return tmpl.substitute(
        caption=caption, label=label, prec=PRECISION,
        colspec=colspec, cols=cols, cmid=cmid, sub=sub
    )

def table_footer():
    # No placeholders -> plain string is fine
    return r"""\bottomrule
\end{tabularx}
\end{table*}"""







def generate_table_for_blocks(df, caption, label, blocks):
    tex = [table_common_header(caption, label, noisy_cols_with_delta=False)]
    for title, dims, cats, splits in blocks:
        tex.append(fr"\multicolumn{{10}}{{l}}{{\textbf{{{title}}}}} \\")
        rows = block_rows(df, dims, cats, splits, include_delta=False)
        tex.extend(rows)
        tex.append(r"\midrule")
    # Remove final \midrule
    if tex[-1] == r"\midrule": tex.pop()
    tex.append(table_footer())
    return "\n".join(tex)

def generate_noisy_delta_table(df, caption, label, blocks):
    tex = [table_common_header(caption, label, noisy_cols_with_delta=True)]
    for title, dims, cats, splits in blocks:
        tex.append(fr"\multicolumn{{14}}{{l}}{{\textbf{{{title}}}}} \\")
        rows = block_rows(df, dims, cats, splits, include_delta=True)
        tex.extend(rows)
        tex.append(r"\midrule")
    if tex[-1] == r"\midrule": tex.pop()
    tex.append(table_footer())
    return "\n".join(tex)

def generate_all_tables(df: pd.DataFrame):
    t1 = generate_table_for_blocks(
        df,
        caption="1D systems (Full: 10 cats; Small: 4 cats). Accuracies (\\%) for Koopman vs. Basic, Strict-/Greedy-H, under Clean and Noisy data. Each block lists hash sizes (4,6,8) and both train/test splits. Uses combined Hungarian metrics.",
        label="tab:results_1d_all",
        blocks=BLOCKS_1D,
    )
    t2 = generate_table_for_blocks(
        df,
        caption="3D systems (Full: 7 cats; Small: 4 cats). Accuracies (\\%) for Koopman vs. Basic, Strict-/Greedy-H, under Clean and Noisy data. Each block lists hash sizes (4,6,8) and both train/test splits. Uses combined Hungarian metrics.",
        label="tab:results_3d_all",
        blocks=BLOCKS_3D,
    )
    # Smaller-scale (Small subsets only, 1D + 3D)
    blocks_small_only = [b for b in BLOCKS_1D + BLOCKS_3D if "Small" in b[0]]
    t3 = generate_table_for_blocks(
        df,
        caption="Smaller-scale experiments (Small subsets only; 1D and 3D). Accuracies (\\%) for Koopman vs. Basic, Strict-/Greedy-H, under Clean and Noisy data across hash sizes and splits. Uses combined Hungarian metrics.",
        label="tab:results_small_only",
        blocks=blocks_small_only,
    )
    # Noisy comparison with deltas across all groups
    t4 = generate_noisy_delta_table(
        df,
        caption="Clean vs. Noisy comparison with absolute change $\\Delta$ (Noisy $-$ Clean, percentage points) across all groups. Uses combined Hungarian metrics.",
        label="tab:results_noisy_comparison",
        blocks=BLOCKS_1D + BLOCKS_3D,
    )
    return {
        "results_1d_all": t1,
        "results_3d_all": t2,
        "results_small_only": t3,
        "results_noisy_comparison": t4,
    }

def main():
    if CSV_PATH:
        df = pd.read_csv(CSV_PATH)
    else:
        raise SystemExit("Set CSV_PATH to your CSV or call generate_all_tables(df) with your in-memory DataFrame.")
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
    tables = generate_all_tables(df)
    for name, tex in tables.items():
        outp = Path(OUT_DIR) / f"{name}.tex"
        outp.write_text(tex, encoding="utf-8")
        print(f"Wrote {outp}")

if __name__ == "__main__":
    main()
