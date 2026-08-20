"""
ips/ips_generate_latex.py
--------------------------------
Standalone, READ-ONLY LaTeX table-generation script. Reads already-produced
CSVs from results/inventory_penalty_sensitivity/ (policy_summary.csv,
comparison_original_vs_high_penalty.csv, both written by
ips_aggregate_summary.py) and writes a single paste-ready
inventory_penalty_sensitivity.tex. Never trains, evaluates, or aggregates
anything; never hard-codes a numerical result -- every number in the output
comes from a CSV cell. Contains ONLY the experimental setup and numerical
values -- no interpretation, no conclusions, no significance language,
matching the brief's explicit instruction.

Run from repo root:
    python ips/ips_generate_latex.py
"""
import sys as _sys
from pathlib import Path as _Path
_REPO_ROOT = _Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_REPO_ROOT))
del _sys, _Path, _REPO_ROOT

import numpy as np
import pandas as pd

import ips.ips_common as IC

POLICY_LABELS = {
    "hamilton_ppo": "Hamilton belief-state MLP",
    "return_mlp_ppo": "Raw-return MLP",
    "return_lstm_ppo": "Raw-return LSTM",
    "oracle": "Oracle",
    "belief_weighted": "Belief-weighted",
    "frozen_clone": "Supervised clone",
}
POLICY_ORDER = ["hamilton_ppo", "return_mlp_ppo", "return_lstm_ppo", "oracle", "belief_weighted", "frozen_clone"]
PPO_POLICIES = {"hamilton_ppo", "return_mlp_ppo", "return_lstm_ppo"}

F = "%.3f"


def fnum(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "--"
    return F % v


def esc(s: str) -> str:
    return str(s).replace("_", r"\_")


def load_data():
    policy_path = IC.RESULTS_DIR / "policy_summary.csv"
    comparison_path = IC.RESULTS_DIR / "comparison_original_vs_high_penalty.csv"
    assert policy_path.exists(), f"{policy_path} not found -- run ips_aggregate_summary.py first"
    policy_df = pd.read_csv(policy_path)
    comparison_df = pd.read_csv(comparison_path) if comparison_path.exists() else None
    return policy_df, comparison_df


def build_principal_table(policy_df: pd.DataFrame) -> str:
    """Higher-penalty (phi=0.10, alpha=0.010) inventory results for every
    PPO and reference policy."""
    sub = policy_df[policy_df["penalty_calibration"] == "high_penalty"].set_index("policy")

    cols = [
        ("cross_seed_mean_time_avg_abs_inventory", "cross_seed_sd_time_avg_abs_inventory", r"Mean $|Q|$"),
        ("cross_seed_mean_integrated_squared_inventory", "cross_seed_sd_integrated_squared_inventory",
         r"$\int Q^2\,dt$"),
        ("cross_seed_mean_rms_inventory", "cross_seed_sd_rms_inventory", r"RMS $Q$"),
        ("cross_seed_mean_max_abs_inventory", "cross_seed_sd_max_abs_inventory", r"Max $|Q|$"),
        ("cross_seed_mean_abs_terminal_inventory", "cross_seed_sd_abs_terminal_inventory", r"$|Q_T|$"),
        ("cross_seed_mean_squared_terminal_inventory", "cross_seed_sd_squared_terminal_inventory", r"$Q_T^2$"),
        ("cross_seed_mean_running_penalty_contribution", "cross_seed_sd_running_penalty_contribution",
         r"Running penalty"),
        ("cross_seed_mean_terminal_penalty_contribution", "cross_seed_sd_terminal_penalty_contribution",
         r"Terminal penalty"),
    ]

    lines = [
        r"\begin{table}[ht]", r"\centering", r"\small",
        r"\begin{tabular}{l" + "r" * len(cols) + "}", r"\toprule",
        "Policy & " + " & ".join(c[2] for c in cols) + r" \\", r"\midrule",
    ]
    for policy in POLICY_ORDER:
        if policy not in sub.index:
            continue
        row = sub.loc[policy]
        cells = [esc(POLICY_LABELS[policy])]
        is_ppo = policy in PPO_POLICIES
        for mean_col, sd_col, _ in cols:
            m = fnum(row.get(mean_col))
            if is_ppo:
                s = fnum(row.get(sd_col))
                cells.append(f"{m} ({s})")
            else:
                cells.append(m)
        lines.append(" & ".join(cells) + r" \\")
    lines += [
        r"\bottomrule", r"\end{tabular}",
        r"\caption{Higher-penalty calibration ($\phi=0.10$, $\alpha=0.010$) holdout inventory results, "
        r"500 holdout paths per policy. PPO rows report the cross-seed mean over 5 learner seeds, with the "
        r"cross-seed standard deviation in parentheses; reference-policy rows (Oracle, Belief-weighted, "
        r"Supervised clone) have no learner-seed dimension.}",
        r"\label{tab:ips-high-penalty-inventory}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def build_comparison_table(policy_df: pd.DataFrame) -> str:
    """Compact original- vs. higher-penalty inventory-value comparison,
    one row per policy (PPO rows use the cross-seed mean over 5 learner
    seeds; reference-policy rows have no learner-seed dimension)."""
    piv_orig = policy_df[policy_df["penalty_calibration"] == "original"].set_index("policy")
    piv_high = policy_df[policy_df["penalty_calibration"] == "high_penalty"].set_index("policy")

    cols = [
        ("cross_seed_mean_time_avg_abs_inventory", r"Mean $|Q|$"),
        ("cross_seed_mean_integrated_squared_inventory", r"$\int Q^2\,dt$"),
        ("cross_seed_mean_rms_inventory", r"RMS $Q$"),
        ("cross_seed_mean_running_penalty_contribution", r"Running penalty"),
        ("cross_seed_mean_terminal_penalty_contribution", r"Terminal penalty"),
    ]

    lines = [
        r"\begin{table}[ht]", r"\centering", r"\small",
        r"\begin{tabular}{ll" + "r" * len(cols) + "}", r"\toprule",
        r"Policy & Calibration & " + " & ".join(c[1] for c in cols) + r" \\", r"\midrule",
    ]
    for policy in POLICY_ORDER:
        if policy not in piv_orig.index or policy not in piv_high.index:
            continue
        for calib_label, piv in ((r"$\phi=0.01,\alpha=0.001$", piv_orig), (r"$\phi=0.10,\alpha=0.010$", piv_high)):
            row = piv.loc[policy]
            cells = [esc(POLICY_LABELS[policy]) if calib_label.startswith(r"$\phi=0.01") else "",
                     calib_label]
            for col, _ in cols:
                cells.append(fnum(row.get(col)))
            lines.append(" & ".join(cells) + r" \\")
        lines.append(r"\addlinespace")
    lines += [
        r"\bottomrule", r"\end{tabular}",
        r"\caption{Original- vs.\ higher-penalty inventory values by policy (cross-seed mean for the 3 PPO "
        r"architectures over 5 learner seeds; single evaluation for the 3 reference policies), 500 holdout "
        r"paths, identical exogenous paths under both calibrations.}",
        r"\label{tab:ips-original-vs-high-penalty}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def build_paired_diff_table(comparison_df: pd.DataFrame) -> str:
    """Paired (same 500 holdout paths) high-penalty-minus-original
    differences with standard errors and 95% CIs, restricted to the
    'inventory_behaviour' metric group and to the fixed 1,000,000-transition
    PPO checkpoints -- cross-seed-averaged for compactness (per-seed rows
    remain available in the CSV)."""
    beh = comparison_df[comparison_df["metric_group"] == "inventory_behaviour"].copy()
    ppo = beh[beh["policy"].isin(PPO_POLICIES)]
    ref = beh[~beh["policy"].isin(PPO_POLICIES)]

    ppo_agg = (ppo.groupby(["policy", "metric"])
               .agg(mean_diff=("mean_diff_high_minus_original", "mean"),
                    se_diff=("mean_diff_high_minus_original", lambda s: float(np.std(s, ddof=1) / np.sqrt(len(s)))
                             if len(s) > 1 else np.nan),
                    n_seeds=("learner_seed", "nunique"))
               .reset_index())
    ref_agg = ref[["policy", "metric", "mean_diff_high_minus_original", "se_diff"]].rename(
        columns={"mean_diff_high_minus_original": "mean_diff"})
    ref_agg["n_seeds"] = 1

    combined = pd.concat([ppo_agg, ref_agg[["policy", "metric", "mean_diff", "se_diff", "n_seeds"]]],
                          ignore_index=True)

    metric_labels = {
        "time_avg_signed_inventory": r"Mean signed $Q$ (time-avg.)",
        "time_avg_abs_inventory": r"Mean $|Q|$ (time-avg.)",
        "integrated_squared_inventory": r"$\int Q^2\,dt$",
        "rms_inventory": r"RMS $Q$",
        "max_abs_inventory": r"Max $|Q|$",
        "terminal_inventory": r"$Q_T$",
        "abs_terminal_inventory": r"$|Q_T|$",
        "squared_terminal_inventory": r"$Q_T^2$",
        "total_fills": r"Total fills",
    }

    lines = [
        r"\begin{table}[ht]", r"\centering", r"\small",
        r"\begin{tabular}{llrrr}", r"\toprule",
        r"Policy & Metric & Mean diff.\ (high $-$ orig.) & SE & $n$ seeds \\", r"\midrule",
    ]
    for policy in POLICY_ORDER:
        sub = combined[combined["policy"] == policy]
        if len(sub) == 0:
            continue
        for metric, label in metric_labels.items():
            r = sub[sub["metric"] == metric]
            if len(r) == 0:
                continue
            r = r.iloc[0]
            lines.append(f"{esc(POLICY_LABELS[policy])} & {label} & {fnum(r['mean_diff'])} & "
                          f"{fnum(r['se_diff'])} & {int(r['n_seeds'])} " + r"\\")
    lines += [
        r"\bottomrule", r"\end{tabular}",
        r"\caption{Paired high-penalty-minus-original differences in unpenalised inventory-behaviour metrics, "
        r"500 identical holdout paths under both calibrations. For the 3 PPO architectures, per-seed paired "
        r"differences (5 learner seeds, fixed 1,000,000-transition checkpoint) are averaged and the standard "
        r"error is computed across those 5 seed-level differences; reference policies have a single paired "
        r"difference over the 500 paths, with its own path-level standard error.}",
        r"\label{tab:ips-paired-diffs}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def main():
    policy_df, comparison_df = load_data()

    parts = [
        r"\subsection{Inventory-Penalty Sensitivity}",
        r"\label{subsec:inventory-penalty-sensitivity}",
        "",
        r"The running inventory penalty $\phi$ was increased from $0.01$ to $0.10$ and the terminal "
        r"inventory penalty $\alpha$ was increased from $0.001$ to $0.010$, a uniform tenfold increase of "
        r"both parameters that preserves their original relative weighting. All three PPO architectures "
        r"(Hamilton belief-state MLP, raw-return MLP, raw-return LSTM) were retrained from scratch under "
        r"this higher-penalty reward, and the regime-conditioned analytical policy, the belief-weighted "
        r"analytical policy, and the frozen supervised clone were all recomputed against the higher-penalty "
        r"calibration.",
        "",
        r"Every other environment parameter (fill-decay $\kappa$, arrival intensity $\lambda$, adverse-"
        r"selection jump size $\epsilon$, regime volatilities, transition generator, terminal time), every "
        r"PPO/RecurrentPPO hyperparameter (learning rate, rollout length, mini-batch size, number of epochs, "
        r"discount factor, GAE parameter, clipping range, initial policy standard deviation), the network "
        r"architecture, the training budget ($1{,}000{,}000$ observable-event transitions per run), the "
        r"number of learner seeds (5 per architecture), the training-environment seed, and the 500-path "
        r"holdout evaluation set (identical paths/seeds to the original calibration) were held fixed between "
        r"the two calibrations.",
        "",
        build_principal_table(policy_df),
        "",
        build_comparison_table(policy_df),
        "",
    ]
    if comparison_df is not None:
        parts += [build_paired_diff_table(comparison_df), ""]
    else:
        parts += [
            r"% Paired original-vs-high-penalty difference table omitted: "
            r"comparison_original_vs_high_penalty.csv not yet available.",
            "",
        ]

    out_path = IC.REPO_ROOT / "inventory_penalty_sensitivity.tex"
    out_path.write_text("\n".join(parts) + "\n", encoding="utf-8")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
