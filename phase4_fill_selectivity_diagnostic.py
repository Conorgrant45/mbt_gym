"""
phase4_fill_selectivity_diagnostic.py
------------------------------------------
Phase 4, Section 4: converts the action-surface grid (already computed by
phase4_action_surface_diagnostic.py) into expected fill probabilities via
p_fill = exp(-kappa * depth), and compares learned vs. analytical policies
on expected bid/ask/total fill probability, fill imbalance and quoted
spread -- on the COMMON GRID, i.e. holding the state distribution fixed, to
isolate "wider quotes at a given state" from "different state visitation"
(the latter is Section 5's job). Does not infer fill behaviour from mean
spread alone -- computes the actual fill-probability transform.

Run from repo root:
    python phase4_fill_selectivity_diagnostic.py
"""
import numpy as np
import pandas as pd

import phase4_common as P4

KAPPA = P4.KAPPA


def add_fill_probabilities(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["p_fill_bid"] = np.exp(-KAPPA * df["bid_depth"])
    df["p_fill_ask"] = np.exp(-KAPPA * df["ask_depth"])
    df["p_fill_total"] = df["p_fill_bid"] + df["p_fill_ask"]
    df["fill_imbalance"] = df["p_fill_bid"] - df["p_fill_ask"]
    df["quoted_spread"] = df["bid_depth"] + df["ask_depth"]
    return df


def main():
    grid_path = P4.PHASE4_RESULTS_DIR / "phase4_action_surface_grid.csv"
    df = pd.read_csv(grid_path)
    df = add_fill_probabilities(df)

    rows = []
    analytical = df[df.policy == "analytical"]
    analytical_summary = dict(
        policy="analytical",
        mean_p_fill_bid=analytical["p_fill_bid"].mean(), mean_p_fill_ask=analytical["p_fill_ask"].mean(),
        mean_p_fill_total=analytical["p_fill_total"].mean(), mean_fill_imbalance=analytical["fill_imbalance"].mean(),
        mean_quoted_spread=analytical["quoted_spread"].mean(),
    )
    rows.append(analytical_summary)

    for policy in sorted(df.policy.unique()):
        if policy == "analytical":
            continue
        sub = df[df.policy == policy]
        row = dict(
            policy=policy,
            mean_p_fill_bid=sub["p_fill_bid"].mean(), mean_p_fill_ask=sub["p_fill_ask"].mean(),
            mean_p_fill_total=sub["p_fill_total"].mean(), mean_fill_imbalance=sub["fill_imbalance"].mean(),
            mean_quoted_spread=sub["quoted_spread"].mean(),
        )
        # Fill-probability ratio vs analytical AT THE SAME grid states -- isolates
        # "wider quotes at a given state" (this ratio) from "different state
        # visitation" (Section 5), since both policies are evaluated on the
        # identical common grid here.
        row["p_fill_total_ratio_vs_analytical"] = row["mean_p_fill_total"] / analytical_summary["mean_p_fill_total"]
        row["spread_ratio_vs_analytical"] = row["mean_quoted_spread"] / analytical_summary["mean_quoted_spread"]
        # Decompose: if fill probability were driven purely by the spread
        # difference (assuming kappa-exponential decay applies uniformly),
        # log(p_fill_ratio) / log ratio driven by spread alone gives a rough
        # attribution -- report both directly rather than assume the relationship.
        rows.append(row)

    out = pd.DataFrame(rows)
    out_path = P4.PHASE4_RESULTS_DIR / "phase4_fill_selectivity.csv"
    out.to_csv(out_path, index=False)
    print(f"Fill-selectivity summary saved to {out_path}")
    print(out.to_string(index=False))

    print("\nInterpretation: since this compares policies on the IDENTICAL common")
    print("grid (same states), any fill-probability shortfall shown here is")
    print("attributable to WIDER QUOTES AT THOSE STATES, not different state")
    print("visitation -- Section 5's state-visitation diagnostic checks whether")
    print("realised rollouts additionally concentrate in low-fill-probability regions.")


if __name__ == "__main__":
    main()
