"""
plot_oracle_naive_results.py
------------------------------
Standalone plotting script. Reads ONLY results/oracle_naive_paired_1000.csv
-- does not rerun any simulation and does not import any environment,
policy, filter, or optimal-control module. Produces three diagnostic
figures and prints all numbers used to build them.

Run:
    python plot_oracle_naive_results.py
"""

import time
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

t_start = time.time()

SCRIPT_DIR = Path(__file__).resolve().parent
CSV_PATH = SCRIPT_DIR / "results" / "oracle_naive_paired_1000.csv"
IMAGES_DIR = Path(r"C:\Users\conor\OneDrive\Documents\Edinburgh master\Dissertation\Python\Regime-Aware-Reinforcement-Learning\images")

COLOR_ORACLE = "#4C72B0"
COLOR_NAIVE = "#C44E52"
COLOR_POS = "#55A868"
COLOR_NEG = "#C44E52"
INK = "#2b2b2b"
MUTED = "#6b6b6b"

# ======================================================================
# Load data
# ======================================================================
df = pd.read_csv(CSV_PATH)
print(f"CSV rows loaded: {len(df)}")

piv = {
    col: df.pivot(index="episode", columns="policy", values=col)
    for col in [
        "raw_pnl", "full_objective", "spread_revenue", "adverse_selection_loss",
        "running_inventory_penalty", "terminal_inventory_penalty",
        "mean_absolute_inventory", "terminal_absolute_inventory", "total_fills",
    ]
}
n_pairs = piv["raw_pnl"].dropna().shape[0]
print(f"Matched episode pairs: {n_pairs}")
assert n_pairs == 1000, f"Expected 1000 matched pairs, found {n_pairs}"


def paired_diff(col):
    return piv[col]["oracle"] - piv[col]["naive"]


def ci_stats(D):
    mean = np.mean(D)
    se = np.std(D, ddof=1) / np.sqrt(len(D))
    return mean, se, mean - 1.96 * se, mean + 1.96 * se


# ======================================================================
# Figure 1: paired PnL / objective differences (CI plot)
# ======================================================================
D_pnl = paired_diff("raw_pnl")
D_objective = paired_diff("full_objective")

mean_pnl, se_pnl, lo_pnl, hi_pnl = ci_stats(D_pnl)
mean_obj, se_obj, lo_obj, hi_obj = ci_stats(D_objective)

print("\n=== Figure 1: paired differences ===")
print(f"Raw PnL:        mean={mean_pnl:.3f}  se={se_pnl:.3f}  95% CI=[{lo_pnl:.3f}, {hi_pnl:.3f}]")
print(f"Full objective: mean={mean_obj:.3f}  se={se_obj:.3f}  95% CI=[{lo_obj:.3f}, {hi_obj:.3f}]")

fig1, ax = plt.subplots(figsize=(9, 3.6), dpi=150)
rows = [("Full objective", mean_obj, lo_obj, hi_obj), ("Raw PnL", mean_pnl, lo_pnl, hi_pnl)]
y_pos = np.arange(len(rows))

for y, (label, mean, lo, hi) in zip(y_pos, rows):
    ax.plot([lo, hi], [y, y], color=COLOR_ORACLE, linewidth=2, solid_capstyle="round", zorder=2)
    ax.plot([lo, lo], [y - 0.06, y + 0.06], color=COLOR_ORACLE, linewidth=2, zorder=2)
    ax.plot([hi, hi], [y - 0.06, y + 0.06], color=COLOR_ORACLE, linewidth=2, zorder=2)
    ax.scatter([mean], [y], color=COLOR_ORACLE, s=70, zorder=3, edgecolor="white", linewidth=0.8)
    ax.annotate(f"mean={mean:.3f}   95% CI=[{lo:.3f}, {hi:.3f}]",
                xy=(hi, y), xytext=(8, 0), textcoords="offset points",
                va="center", ha="left", fontsize=9.5, color=INK)

ax.axvline(0, color=MUTED, linestyle="--", linewidth=1.2, zorder=1)
ax.set_yticks(y_pos)
ax.set_yticklabels([r[0] for r in rows], fontsize=11)
ax.set_ylim(-0.6, len(rows) - 0.4)
ax.set_xlim(-1.5, max(hi_pnl, hi_obj) + 6.5)
ax.set_xlabel("Mean paired difference: oracle − naive", fontsize=10.5, color=INK)
for spine in ["top", "right", "left"]:
    ax.spines[spine].set_visible(False)
ax.spines["bottom"].set_color("#b0b0b0")
ax.tick_params(left=False, colors=MUTED)
ax.grid(axis="x", color="#e3e3e3", linewidth=0.8, zorder=0)
ax.set_axisbelow(True)
fig1.text(0.5, 0.985, "Oracle advantage over naive policy", ha="center", va="top",
           fontsize=14, color=INK, fontweight="bold")
fig1.text(0.5, 0.93, "Points show mean paired differences across 1,000 episodes; "
           "error bars show 95% confidence intervals.", ha="center", va="top",
           fontsize=9.5, color=MUTED)
plt.tight_layout(rect=[0, 0, 1, 0.87])

fig1_path = IMAGES_DIR / "oracle_naive_pnl_objective_ci.png"
fig1.savefig(fig1_path, dpi=200, bbox_inches="tight", facecolor="white")
plt.close(fig1)


# ======================================================================
# Figure 2: economic decomposition (waterfall)
# ======================================================================
D_spread = paired_diff("spread_revenue")
D_adverse_benefit = piv["adverse_selection_loss"]["naive"] - piv["adverse_selection_loss"]["oracle"]
D_running_penalty_benefit = piv["running_inventory_penalty"]["naive"] - piv["running_inventory_penalty"]["oracle"]
D_terminal_penalty_benefit = piv["terminal_inventory_penalty"]["naive"] - piv["terminal_inventory_penalty"]["oracle"]

mean_spread = np.mean(D_spread)
mean_adverse_benefit = np.mean(D_adverse_benefit)
mean_running_benefit = np.mean(D_running_penalty_benefit)
mean_terminal_benefit = np.mean(D_terminal_penalty_benefit)

mean_residual = mean_obj - mean_spread - mean_adverse_benefit - mean_running_benefit - mean_terminal_benefit

print("\n=== Figure 2: decomposition ===")
print(f"1. Lost spread revenue            = {mean_spread:.4f}")
print(f"2. Reduced adverse-selection loss  = {mean_adverse_benefit:.4f}")
print(f"3. Other PnL effects (residual)    = {mean_residual:.4f}")
print(f"4. Lower running penalty           = {mean_running_benefit:.4f}")
print(f"5. Lower terminal penalty          = {mean_terminal_benefit:.4f}")
reconciled_total = mean_spread + mean_adverse_benefit + mean_residual + mean_running_benefit + mean_terminal_benefit
print(f"6. Net objective advantage (sum)   = {reconciled_total:.4f}   (mean(D_objective) = {mean_obj:.4f})")
recon_err = abs(reconciled_total - mean_obj)
print(f"Reconciliation error: {recon_err:.3e}")
assert recon_err < 1e-6, "Decomposition does not reconcile with mean(D_objective)"

labels = ["Lost spread\nrevenue", "Reduced adverse-\nselection loss", "Other PnL\neffects",
          "Lower running\npenalty", "Lower terminal\npenalty", "Net objective\nadvantage"]
values = [mean_spread, mean_adverse_benefit, mean_residual, mean_running_benefit, mean_terminal_benefit, mean_obj]

fig2, ax = plt.subplots(figsize=(11, 6), dpi=150)
cumulative = 0.0
x = np.arange(len(labels))
bar_width = 0.6

for i, (label, val) in enumerate(zip(labels, values)):
    if i < 5:  # floating step bars
        bottom = cumulative
        top = cumulative + val
        color = COLOR_POS if val >= 0 else COLOR_NEG
        ax.bar(x[i], val, bottom=bottom, width=bar_width, color=color, alpha=0.9, zorder=3)
        y_annot = max(bottom, top) + 0.15
        ax.annotate(f"{val:+.3f}", xy=(x[i], y_annot), ha="center", va="bottom",
                    fontsize=9.5, color=INK, zorder=4)
        # connector line to next bar's start
        if i < 4:
            ax.plot([x[i] + bar_width / 2, x[i + 1] - bar_width / 2], [top, top],
                     color="#b0b0b0", linewidth=1, linestyle=":", zorder=2)
        cumulative = top
    else:  # final total bar, from 0
        ax.bar(x[i], val, bottom=0, width=bar_width, color=COLOR_ORACLE, alpha=0.95, zorder=3)
        ax.annotate(f"{val:+.3f}", xy=(x[i], val + 0.15), ha="center", va="bottom",
                    fontsize=9.5, color=INK, fontweight="bold", zorder=4)

ax.axhline(0, color="#b0b0b0", linewidth=1)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=9.5)
ax.set_ylabel("Oracle − naive contribution", fontsize=10.5, color=INK)
ax.grid(axis="y", color="#e3e3e3", linewidth=0.8, zorder=0)
ax.set_axisbelow(True)
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
for spine in ["left", "bottom"]:
    ax.spines[spine].set_color("#b0b0b0")
ax.tick_params(colors=MUTED)
fig2.text(0.5, 0.98, "Decomposition of the oracle's objective advantage", ha="center", va="top",
           fontsize=14, color=INK, fontweight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.93])

fig2_path = IMAGES_DIR / "oracle_naive_objective_decomposition.png"
fig2.savefig(fig2_path, dpi=200, bbox_inches="tight", facecolor="white")
plt.close(fig2)


# ======================================================================
# Figure 3: trading activity and inventory exposure (relative to naive)
# ======================================================================
metrics = ["adverse_selection_loss", "total_fills", "mean_absolute_inventory", "terminal_absolute_inventory"]
metric_labels = ["Adverse-selection\nloss", "Total fills", "Mean absolute\ninventory", "Terminal absolute\ninventory"]

policy_means = {m: dict(oracle=piv[m]["oracle"].mean(), naive=piv[m]["naive"].mean()) for m in metrics}

print("\n=== Figure 3: policy-level means and relative values ===")
relative_values = []
for m, lbl in zip(metrics, metric_labels):
    o_mean = policy_means[m]["oracle"]
    n_mean = policy_means[m]["naive"]
    rel = 100 * o_mean / n_mean
    reduction = 100 - rel
    relative_values.append(rel)
    print(f"{m:<28} oracle_mean={o_mean:.4f}  naive_mean={n_mean:.4f}  "
          f"oracle_pct_of_naive={rel:.1f}%  reduction={reduction:.1f}%")

fig3, ax = plt.subplots(figsize=(9.5, 5.5), dpi=150)
x = np.arange(len(metrics))
bars = ax.bar(x, relative_values, width=0.55, color=COLOR_ORACLE, alpha=0.9, zorder=3)
ax.axhline(100, color=COLOR_NAIVE, linestyle="--", linewidth=1.4, zorder=2, label="Naive benchmark (100%)")

for xi, rel in zip(x, relative_values):
    reduction = 100 - rel
    ax.annotate(f"{rel:.1f}% of naive\n({reduction:.1f}% lower)", xy=(xi, rel), xytext=(0, 6),
                textcoords="offset points", ha="center", va="bottom", fontsize=9.5, color=INK)

ax.set_xticks(x)
ax.set_xticklabels(metric_labels, fontsize=10)
ax.set_ylabel("Oracle as percentage of naive", fontsize=10.5, color=INK)
ax.set_ylim(0, 115)
ax.legend(fontsize=9.5, frameon=False, loc="lower right")
ax.grid(axis="y", color="#e3e3e3", linewidth=0.8, zorder=0)
ax.set_axisbelow(True)
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
for spine in ["left", "bottom"]:
    ax.spines[spine].set_color("#b0b0b0")
ax.tick_params(colors=MUTED)
fig3.text(0.5, 0.98, "Oracle trading activity and inventory exposure relative to naive",
           ha="center", va="top", fontsize=13.5, color=INK, fontweight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.92])

fig3_path = IMAGES_DIR / "oracle_naive_activity_inventory.png"
fig3.savefig(fig3_path, dpi=200, bbox_inches="tight", facecolor="white")
plt.close(fig3)


# ======================================================================
# Summary
# ======================================================================
elapsed = time.time() - t_start
print("\n=== Saved figures ===")
print(fig1_path)
print(fig2_path)
print(fig3_path)
print(f"\nTotal script runtime: {elapsed:.2f}s")
