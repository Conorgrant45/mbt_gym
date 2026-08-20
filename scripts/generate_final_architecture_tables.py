"""
scripts/generate_final_architecture_tables.py
--------------------------------------------------
Standalone, READ-ONLY table-generation script for the final reduced-
exploration architecture-comparison experiment. Reads already-produced
CSVs from --results-dir and regenerates Markdown + LaTeX (booktabs) tables;
never trains or evaluates a model; never hard-codes a numerical result
(every number comes from a CSV cell).

Run from repo root (PowerShell), e.g.:

    python scripts/generate_final_architecture_tables.py `
        --results-dir results/final_reduced_exploration_architecture_comparison `
        --output-dir results/final_reduced_exploration_architecture_comparison/tables
"""
import argparse
from pathlib import Path

import pandas as pd

ARCHITECTURES = ("hamilton_ppo", "return_mlp_ppo", "return_lstm_ppo")
ARCHITECTURE_LABELS = {"hamilton_ppo": "Hamilton belief-state MLP", "return_mlp_ppo": "Raw-return MLP",
                        "return_lstm_ppo": "Raw-return LSTM"}

FINAL_COMPARISON_METRICS = [
    ("cross_seed_mean_full_objective", "cross_seed_sd_full_objective", "Full objective"),
    ("cross_seed_mean_raw_pnl", None, "Raw PnL"),
    ("cross_seed_mean_mean_fills", None, "Fills"),
    ("cross_seed_mean_total_arrivals", None, "Arrivals"),
    ("cross_seed_mean_fill_to_arrival_ratio", None, "Fill-to-arrival ratio"),
    ("cross_seed_mean_mean_quoted_spread", None, "Mean quoted spread"),
    ("cross_seed_mean_mean_bid_depth", None, "Mean bid depth"),
    ("cross_seed_mean_mean_ask_depth", None, "Mean ask depth"),
    ("cross_seed_mean_spread_revenue", None, "Spread revenue"),
    ("cross_seed_mean_spread_revenue_per_fill", None, "Spread revenue per fill"),
    ("cross_seed_mean_adverse_selection_loss", None, "Adverse-selection loss"),
    ("cross_seed_mean_running_penalty", None, "Running inventory penalty"),
    ("cross_seed_mean_terminal_penalty", None, "Terminal inventory penalty"),
    ("cross_seed_mean_mean_abs_inventory", None, "Mean |inventory|"),
    ("cross_seed_mean_action_std_bid", None, "Learned action std (bid)"),
    ("cross_seed_mean_action_std_ask", None, "Learned action std (ask)"),
    ("cross_seed_mean_near_bound_rate_deterministic", None, "Deterministic near-bound rate"),
    ("cross_seed_mean_near_bound_rate_stochastic", None, "Stochastic near-bound rate"),
]


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-dir", required=True, type=Path)
    p.add_argument("--output-dir", type=Path, default=None, help="Default: <results-dir>/tables")
    p.add_argument("--float-format", type=str, default="%.4f")
    return p


def load_data(results_dir: Path) -> dict:
    def _read(name, required=True):
        p = results_dir / name
        if not p.exists():
            if required:
                raise FileNotFoundError(f"Required file not found: {p}")
            return None
        return pd.read_csv(p)

    return dict(
        validation_arch=_read("validation_architecture_summary.csv"),
        holdout_arch=_read("final_holdout_architecture_summary.csv"),
        fixed_200k_1m=_read("fixed_200k_vs_1m_contrasts.csv"),
        benchmark_paired=_read("benchmark_paired_contrasts.csv"),
        architecture_contrasts=_read("architecture_contrasts.csv"),
    )


def _fmt(v, float_format: str) -> str:
    if isinstance(v, float):
        return float_format % v
    return str(v)


def write_markdown_table(df: pd.DataFrame, path: Path, title: str, float_format: str):
    lines = [f"# {title}", ""]
    cols = list(df.columns)
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "|".join(["---"] * len(cols)) + "|")
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(_fmt(row[c], float_format) for c in cols) + " |")
    path.write_text("\n".join(lines) + "\n")
    print(f"Saved {path}")


def write_latex_table(df: pd.DataFrame, path: Path, caption: str, label: str, float_format: str):
    cols = list(df.columns)
    lines = [
        r"\begin{table}[ht]", r"\centering",
        r"\begin{tabular}{" + "l" * len(cols) + "}", r"\toprule",
        " & ".join(c.replace("_", r"\_") for c in cols) + r" \\", r"\midrule",
    ]
    for _, row in df.iterrows():
        lines.append(" & ".join(_fmt(row[c], float_format) for c in cols) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", rf"\caption{{{caption}}}", rf"\label{{{label}}}", r"\end{table}"]
    path.write_text("\n".join(lines) + "\n")
    print(f"Saved {path}")


# ======================================================================
# Table builders (pure reshape of the loaded CSVs, no new computation)
# ======================================================================
def build_final_comparison_table(holdout_arch: pd.DataFrame) -> pd.DataFrame:
    sub = holdout_arch[holdout_arch["checkpoint_transition"] == 1_000_000].set_index("architecture")
    rows = []
    for architecture in ARCHITECTURES:
        if architecture not in sub.index:
            continue
        row = {"Architecture": ARCHITECTURE_LABELS[architecture]}
        for col, err_col, label in FINAL_COMPARISON_METRICS:
            if col not in sub.columns:
                continue
            val = sub.loc[architecture, col]
            if err_col and err_col in sub.columns:
                row[label] = f"{val:.4f} +/- {sub.loc[architecture, err_col]:.4f}"
            else:
                row[label] = val
        rows.append(row)
    return pd.DataFrame(rows)


def build_learning_curve_checkpoint_table(validation_arch: pd.DataFrame) -> pd.DataFrame:
    keep = ["architecture", "checkpoint_transition", "cross_seed_mean_full_objective",
            "cross_seed_sd_full_objective", "cross_seed_se_full_objective", "n_seeds"]
    keep = [c for c in keep if c in validation_arch.columns]
    return validation_arch[keep].sort_values(["architecture", "checkpoint_transition"]).reset_index(drop=True)


def build_architecture_contrast_table(contrasts: pd.DataFrame) -> pd.DataFrame:
    sub = contrasts[contrasts["metric"] == "full_objective"].copy()
    sub["Contrast"] = sub["architecture_a"].map(ARCHITECTURE_LABELS) + " - " + sub["architecture_b"].map(ARCHITECTURE_LABELS)
    keep = ["Contrast", "mean_matched_seed_diff", "se_matched_seed_diff", "bootstrap_ci_lo", "bootstrap_ci_hi",
            "n_seeds_favouring_a", "n_seeds_favouring_b"]
    return sub[keep].reset_index(drop=True)


def build_benchmark_paired_table(benchmark_paired: pd.DataFrame) -> pd.DataFrame:
    df = benchmark_paired.copy()
    df["Architecture"] = df["architecture"].map(ARCHITECTURE_LABELS)
    keep = ["Architecture", "learner_seed", "benchmark", "paired_mean_diff", "se_diff", "normal_ci_lo",
            "normal_ci_hi", "bootstrap_ci_lo", "bootstrap_ci_hi", "win_rate"]
    return df[keep].sort_values(["Architecture", "learner_seed", "benchmark"]).reset_index(drop=True)


def build_fixed_200k_1m_table(fixed_df: pd.DataFrame) -> pd.DataFrame:
    df = fixed_df.copy()
    df["Architecture"] = df["architecture"].map(ARCHITECTURE_LABELS)
    keep = ["Architecture", "learner_seed", "metric", "mean_change_200k_to_1m", "se_change", "normal_ci_lo",
            "normal_ci_hi", "bootstrap_ci_lo", "bootstrap_ci_hi", "win_rate_1m_better"]
    return df[keep].sort_values(["Architecture", "learner_seed", "metric"]).reset_index(drop=True)


def main():
    args = build_arg_parser().parse_args()
    output_dir = args.output_dir if args.output_dir is not None else args.results_dir / "tables"
    output_dir.mkdir(parents=True, exist_ok=True)
    data = load_data(args.results_dir)

    tables = [
        ("final_architecture_comparison", build_final_comparison_table(data["holdout_arch"]),
         "Final fixed-1,000,000-transition architecture comparison", "tab:final-comparison"),
        ("learning_curve_checkpoints", build_learning_curve_checkpoint_table(data["validation_arch"]),
         "Validation-objective learning curve, every checkpoint", "tab:learning-curve"),
        ("architecture_contrasts", build_architecture_contrast_table(data["architecture_contrasts"]),
         "Matched-learner-seed architecture contrasts (full objective)", "tab:architecture-contrasts"),
        ("benchmark_paired_contrasts", build_benchmark_paired_table(data["benchmark_paired"]),
         "Paired path-level contrasts vs. benchmarks (full objective)", "tab:benchmark-contrasts"),
        ("fixed_200k_vs_1m", build_fixed_200k_1m_table(data["fixed_200k_1m"]),
         "Fixed 200,000- vs. 1,000,000-transition paired contrasts", "tab:200k-vs-1m"),
    ]

    for stem, df, caption, label in tables:
        write_markdown_table(df, output_dir / f"{stem}.md", caption, args.float_format)
        write_latex_table(df, output_dir / f"{stem}.tex", caption, label, args.float_format)

    print(f"\nAll tables written to {output_dir}")


if __name__ == "__main__":
    main()
