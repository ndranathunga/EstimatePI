#!/usr/bin/env python3
import pandas as pd

CSV_FILE = "../results/results_2025-02-23_17-22-09.csv"

ACTUAL_PI = 3.14159265358979323846

# Read columns as strings so decimal values are not rounded
df = pd.read_csv(
    CSV_FILE,
    dtype={
        "Group": str,
        "Experiment": str,
        "Backend": str,
        "ThreadCount": str,
        "Precision": str,
        "TotalSamples": str,
        "PiEstimate": str,
        "TimeSeconds": str
    }
)

df["PiEstimateFloat"] = pd.to_numeric(df["PiEstimate"], errors="raise")
df["Error"] = (df["PiEstimateFloat"] - ACTUAL_PI).abs()

def make_latex_table(
    data: pd.DataFrame,
    columns: list,
    header: list,
    caption: str,
    label: str,
    filename: str,
    column_format: str = "|l|l|l|"
):
    # Convert to LaTeX with the desired format
    table_str = data.to_latex(
        index=False,
        columns=columns,
        header=header,
        caption=caption,
        label=label,
        escape=False,
        column_format=column_format,
        position="H"
    )

    # Replace the booktabs lines with \hline for a fully bordered grid
    table_str = (
        table_str.replace("\\toprule", "\\hline")
                 .replace("\\midrule", "\\hline")
                 .replace("\\bottomrule", "\\hline")
    )
    
    # add `centering` after [ht] to center the table
    table_str = table_str.replace("\\begin{table}[H]", "\\begin{table}[H]\n\\centering")

    with open(filename, "w") as f:
        f.write(table_str)

# ------------------------------------------------------------------------------
# QUESTION 1: Precision-based
# ------------------------------------------------------------------------------
# 1) Pthreads + "precision" in 'Experiment'
mask_pthreads_precision = (
    df["Experiment"].str.contains("precision", case=False) &
    df["Backend"].str.contains("Pthreads", case=False)
)
df_pthreads_precision = df.loc[mask_pthreads_precision].copy()

make_latex_table(
    data=df_pthreads_precision,
    columns=["Precision", "PiEstimate", "TimeSeconds"],
    header=["Precision", "Pi Estimate", "Time (s)"],
    caption="Precision-based results for Pthreads",
    label="tab:pthreads-precision",
    filename="pthreads_precision_table.tex",
    column_format="|l|l|l|"
)

# 2) OpenMP + "precision"
mask_openmp_precision = (
    df["Experiment"].str.contains("precision", case=False) &
    df["Backend"].str.contains("OpenMP", case=False)
)
df_openmp_precision = df.loc[mask_openmp_precision].copy()

make_latex_table(
    data=df_openmp_precision,
    columns=["Precision", "PiEstimate", "TimeSeconds"],
    header=["Precision", "Pi Estimate", "Time (s)"],
    caption="Precision-based results for OpenMP",
    label="tab:openmp-precision",
    filename="openmp_precision_table.tex",
    column_format="|l|l|l|"
)

# 3) CUDA + "precision"
mask_cuda_precision = (
    df["Experiment"].str.contains("precision", case=False) &
    df["Backend"].str.contains("CUDA", case=False)
)
df_cuda_precision = df.loc[mask_cuda_precision].copy()

make_latex_table(
    data=df_cuda_precision,
    columns=["Precision", "PiEstimate", "TimeSeconds"],
    header=["Precision", "Pi Estimate", "Time (s)"],
    caption="Precision-based results for CUDA",
    label="tab:cuda-precision",
    filename="cuda_precision_table.tex",
    column_format="|l|l|l|"
)

# ------------------------------------------------------------------------------
# QUESTION 2: 2^n-based
# ------------------------------------------------------------------------------
# 4) Pthreads + "2^"
mask_pthreads_2n = (
    df["Experiment"].str.contains("2\\^", case=False, regex=True) &
    df["Backend"].str.contains("Pthreads", case=False)
)
df_pthreads_2n = df.loc[mask_pthreads_2n].copy()

# convert the Experiment from `2^n trials` to `$2^{n}$ trials`
df_pthreads_2n["Experiment"] = df_pthreads_2n["Experiment"].str.replace("2\\^", "$2^{", regex=True)
df_pthreads_2n["Experiment"] = df_pthreads_2n["Experiment"].str.replace(" trials", "}$ trials", regex=False)

make_latex_table(
    data=df_pthreads_2n,
    columns=["Experiment", "TotalSamples", "PiEstimate", "TimeSeconds", "Error"],
    header=["Experiment", "Total Samples", "Pi Estimate", "Time (s)", "Error"],
    caption="$2^n$-based results for Pthreads",
    label="tab:pthreads-2n",
    filename="pthreads_2n_table.tex",
    column_format="|l|l|l|l|l|"
)

# 5) OpenMP + "2^"
mask_openmp_2n = (
    df["Experiment"].str.contains("2\\^", case=False, regex=True) &
    df["Backend"].str.contains("OpenMP", case=False)
)
df_openmp_2n = df.loc[mask_openmp_2n].copy()

# convert the Experiment from `2^n trials` to `$2^{n}$ trials`
df_openmp_2n["Experiment"] = df_openmp_2n["Experiment"].str.replace("2\\^", "$2^{", regex=True)
df_openmp_2n["Experiment"] = df_openmp_2n["Experiment"].str.replace(" trials", "}$ trials", regex=False)

make_latex_table(
    data=df_openmp_2n,
    columns=["Experiment", "TotalSamples", "PiEstimate", "TimeSeconds", "Error"],
    header=["Experiment", "Total Samples", "Pi Estimate", "Time (s)", "Error"],
    caption="$2^n$-based results for OpenMP",
    label="tab:openmp-2n",
    filename="openmp_2n_table.tex",
    column_format="|l|l|l|l|l|"
)

# 6) CUDA + "2^"
mask_cuda_2n = (
    df["Experiment"].str.contains("2\\^", case=False, regex=True) &
    df["Backend"].str.contains("CUDA", case=False)
)
df_cuda_2n = df.loc[mask_cuda_2n].copy()

# convert the Experiment from `2^n trials` to `$2^{n}$ trials`
df_cuda_2n["Experiment"] = df_cuda_2n["Experiment"].str.replace("2\\^", "$2^{", regex=True)
df_cuda_2n["Experiment"] = df_cuda_2n["Experiment"].str.replace(" trials", "}$ trials", regex=False)

make_latex_table(
    data=df_cuda_2n,
    columns=["Experiment", "TotalSamples", "PiEstimate", "TimeSeconds", "Error"],
    header=["Experiment", "Total Samples", "Pi Estimate", "Time (s)", "Error"],
    caption="$2^n$-based results for CUDA",
    label="tab:cuda-2n",
    filename="cuda_2n_table.tex",
    column_format="|l|l|l|l|l|"
)

print("All table .tex files have been generated.")
