"""
figA1_filter_calibration.py
--------------------------------
REQUIRES optional job S1 (build_filter_calibration_dataset.py) to have been
run first -- reads filter_calibration_reliability_bins.csv,
filter_calibration_detection_delays.csv, and filter_calibration_summary.json,
none of which exist until then. Not run as part of the default figure set;
see FIGURES_REPORT.md.

Panel (a): reliability diagram -- binned mean predicted posterior vs.
empirical true-regime frequency (~20 bins), with bin counts shown as marker
size/annotation. Panel (b): histogram of per-switch detection delay (in
EVENTS, not wall-clock time) from a true regime switch to the first
subsequent event where the posterior crosses 0.5 toward the new regime.
Brier score quoted in the stats sidecar (not drawn on the plot).
"""
import numpy as np
import pandas as pd
import json

import fig_style as FS

# ======================================================================
# CONFIGURATION
# ======================================================================
FIG_NAME = "figA1_filter_calibration"
INPUT_RELIABILITY_CSV = FS.RESULTS_DIR / "filter_calibration_reliability_bins.csv"
INPUT_DELAYS_CSV = FS.RESULTS_DIR / "filter_calibration_detection_delays.csv"
INPUT_SUMMARY_JSON = FS.RESULTS_DIR / "filter_calibration_summary.json"


def main():
    missing = [p for p in [INPUT_RELIABILITY_CSV, INPUT_DELAYS_CSV, INPUT_SUMMARY_JSON] if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing {missing} -- this figure requires optional job S1 "
            f"(build_filter_calibration_dataset.py) to be run first. That job is NOT run "
            f"automatically (see FIGURES_REPORT.md); run it explicitly, then re-run this script."
        )
    reliability = pd.read_csv(INPUT_RELIABILITY_CSV)
    delays = pd.read_csv(INPUT_DELAYS_CSV)
    summary = json.loads(INPUT_SUMMARY_JSON.read_text())

    fig, (ax_rel, ax_delay) = FS.new_figure(width="full", height_in=3.2, ncols=2, nrows=1)

    valid = reliability.dropna(subset=["mean_predicted", "empirical_frequency"])
    ax_rel.plot([0, 1], [0, 1], color="0.7", linestyle="--", linewidth=0.8, zorder=1)
    sizes = 8 + 40 * (valid["n"] / valid["n"].max())
    ax_rel.scatter(valid["mean_predicted"], valid["empirical_frequency"], s=sizes,
                    color=FS.COLORS["hamilton_ppo"], edgecolors="black", linewidths=0.4, zorder=2)
    ax_rel.set_xlabel("Mean predicted posterior (bin)")
    ax_rel.set_ylabel("Empirical regime-1 frequency")
    ax_rel.set_xlim(0, 1)
    ax_rel.set_ylim(0, 1)
    FS.panel_letter(ax_rel, "a")

    max_delay = int(delays["detection_delay_events"].max())
    ax_delay.hist(delays["detection_delay_events"], bins=np.arange(0, max_delay + 2) - 0.5,
                   color=FS.COLORS["hamilton_ppo"], edgecolor="black", linewidth=0.4)
    ax_delay.set_yscale("log")
    ax_delay.set_xlim(-0.5, 15.5)
    frac_zero = float((delays["detection_delay_events"] == 0).mean())
    frac_le5 = float((delays["detection_delay_events"] <= 5).mean())
    n_exceed_5 = int((delays["detection_delay_events"] > 5).sum())
    # .1f previously rounded 99.95% up to a misleading "100.0%", directly
    # contradicting both the visible tail bars and the max_observed note on
    # the same line -- .2f plus an explicit exceedance count makes all three
    # numbers on this panel mutually consistent.
    ax_delay.text(0.97, 0.95, f"{frac_zero*100:.1f}% detected within 0 events\n"
                                f"{frac_le5*100:.2f}% within 5 events ({n_exceed_5} exceeded)\n"
                                f"(max observed: {max_delay}, axis clipped at 15)",
                   transform=ax_delay.transAxes, ha="right", va="top", fontsize=6.5, color="dimgray")
    ax_delay.set_xlabel("Detection delay (events)")
    ax_delay.set_ylabel("Count (log scale)")
    FS.panel_letter(ax_delay, "b")

    stats = {"figure": FIG_NAME, "brier_score": summary["brier_score"], "n_events": summary["n_events"],
              "n_switches_detected": summary["n_switches_detected"],
              "median_detection_delay_events": float(delays["detection_delay_events"].median()),
              "mean_detection_delay_events": float(delays["detection_delay_events"].mean())}
    FS.save_figure(fig, FIG_NAME, stats)


if __name__ == "__main__":
    main()
