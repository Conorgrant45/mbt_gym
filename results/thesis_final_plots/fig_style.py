"""
fig_style.py
-------------
Shared style module for the fig01-fig08 / figA1-figA3 thesis figure set
(see FIGURES_REPORT.md). Import this from every new figure script in this
directory; do NOT duplicate the palette/rcParams inline the way the
original 01_-06_ scripts do (those predate this module and are left as-is).

Policy display names/order/colours verified against the actual `policy`/
`architecture` column values in final_holdout_episode_level.csv (see
FIGURES_REPORT.md §1) -- the internal strings hamilton_ppo/return_mlp_ppo/
return_lstm_ppo/oracle/belief_weighted/frozen_clone are confirmed correct;
only the DISPLAY strings are the brief's stylistic choice (in particular
"oracle" is renamed "Regime-conditioned benchmark" everywhere in figures/
legends -- the internal column value is untouched).
"""
from pathlib import Path
import json

import matplotlib
import matplotlib.pyplot as plt

# ======================================================================
# Paths
# ======================================================================
PROJECT_ROOT = Path(r"C:\Users\conor\OneDrive\Documents\Edinburgh master\Dissertation\Python\mbt_gym")
RESULTS_DIR = PROJECT_ROOT / "results" / "final_reduced_exploration_architecture_comparison"
THESIS_DIR = PROJECT_ROOT / "results" / "thesis_final_plots"
FIGURES_DIR = THESIS_DIR / "figures"

# ======================================================================
# Policy identity: internal string -> display name, fixed order, colours,
# linestyles, markers. Identical in every figure that uses this module.
# ======================================================================
RL_POLICIES = ["hamilton_ppo", "return_mlp_ppo", "return_lstm_ppo"]
BENCHMARK_POLICIES = ["oracle", "belief_weighted", "frozen_clone"]
ALL_POLICIES = RL_POLICIES + BENCHMARK_POLICIES

DISPLAY_NAMES = {
    "hamilton_ppo": "Belief-state PPO",
    "return_mlp_ppo": "Raw-return MLP PPO",
    "return_lstm_ppo": "Raw-return LSTM PPO",
    "oracle": "Regime-conditioned benchmark",
    "belief_weighted": "Belief-weighted analytical",
    "frozen_clone": "Frozen clone",
}

COLORS = {
    "hamilton_ppo": "#0072B2",
    "return_mlp_ppo": "#E69F00",
    "return_lstm_ppo": "#009E73",
    "oracle": "#000000",
    "belief_weighted": "0.45",   # 45% grey (matplotlib grayscale string, 0=black..1=white)
    "frozen_clone": "0.65",      # 65% grey
}

LINESTYLES = {
    "hamilton_ppo": "-",
    "return_mlp_ppo": "--",
    "return_lstm_ppo": "-.",
    "oracle": ":",
    "belief_weighted": "-",
    "frozen_clone": "--",
}

LINEWIDTHS = {
    "hamilton_ppo": 1.6, "return_mlp_ppo": 1.6, "return_lstm_ppo": 1.6,
    "oracle": 1.4, "belief_weighted": 1.0, "frozen_clone": 1.0,
}

MARKERS = {
    "hamilton_ppo": "o",
    "return_mlp_ppo": "s",
    "return_lstm_ppo": "^",
    "oracle": "*",
    "belief_weighted": "+",
    "frozen_clone": "x",
}


def display(policy: str) -> str:
    if policy not in DISPLAY_NAMES:
        raise KeyError(f"No display name registered for policy {policy!r}. Registered: {list(DISPLAY_NAMES)}")
    return DISPLAY_NAMES[policy]


def verify_policy_set(observed: set, context: str = "") -> None:
    """Call this once per script, right after loading data, with the set of
    policy/architecture values actually found in the CSV. Raises loudly if
    the data contains a policy this style module doesn't know how to draw,
    rather than silently skipping it or falling back to a default colour."""
    unknown = observed - set(ALL_POLICIES)
    if unknown:
        raise ValueError(
            f"Unknown policy value(s) {sorted(unknown)} found in the data{' (' + context + ')' if context else ''} "
            f"-- fig_style.py's DISPLAY_NAMES/COLORS/etc. only cover {ALL_POLICIES}. "
            f"Update fig_style.py's mapping keys (never the display names) before plotting."
        )


# ======================================================================
# Matplotlib rcParams -- house style for every fig0*/figA* script
# ======================================================================
FULL_WIDTH_IN = 6.3
HALF_WIDTH_IN = 3.1

RC = {
    "font.size": 8.5,
    "axes.titlesize": 9,
    "axes.labelsize": 8.5,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5,
    "figure.titlesize": 9,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.7,
    "lines.linewidth": 1.4,
    "figure.constrained_layout.use": True,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,   # embed as real text, not paths, in the PDF
    "ps.fonttype": 42,
}


def apply_style():
    matplotlib.rcParams.update(RC)


def panel_letter(ax, letter: str, x: float = -0.06, y: float = 1.04):
    """(a)/(b)/... panel label, top-left, in place of an in-figure title.
    No figure in this set uses ax.set_title with descriptive text -- see
    house rule in FIGURES_REPORT.md."""
    ax.text(x, y, f"({letter})", transform=ax.transAxes, fontsize=9, fontweight="bold",
            va="bottom", ha="right")


def new_figure(width: str = "full", height_in: float = None, ncols: int = 1, nrows: int = 1, **kwargs):
    """width: 'full' (6.3in) or 'half' (3.1in). height_in: explicit height,
    else a sensible default based on nrows."""
    apply_style()
    w = FULL_WIDTH_IN if width == "full" else HALF_WIDTH_IN
    h = height_in if height_in is not None else 2.2 * nrows + 0.3
    fig, axes = plt.subplots(nrows, ncols, figsize=(w, h), **kwargs)
    return fig, axes


def save_figure(fig, name: str, stats: dict = None, bbox_inches: str = None):
    """Writes figures/<name>.pdf, figures/<name>.png (300dpi), and
    figures/<name>_stats.json (every number a caption might need -- means,
    CIs, win rates, sample sizes, etc; empty dict if genuinely none).
    bbox_inches: pass "tight" to crop excess whitespace at save-time (default
    None preserves each figure's constrained-layout canvas as-is, unchanged
    for every pre-existing figure in this set)."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = FIGURES_DIR / f"{name}.pdf"
    png_path = FIGURES_DIR / f"{name}.png"
    json_path = FIGURES_DIR / f"{name}_stats.json"
    fig.savefig(pdf_path, bbox_inches=bbox_inches)
    fig.savefig(png_path, dpi=300, bbox_inches=bbox_inches)
    json_path.write_text(json.dumps(stats if stats is not None else {}, indent=2, default=_json_default))
    print(f"Saved {pdf_path}")
    print(f"Saved {png_path}")
    print(f"Saved {json_path}")


def _json_default(o):
    import numpy as np
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    if isinstance(o, (Path,)):
        return str(o)
    raise TypeError(f"Object of type {type(o)} is not JSON serialisable: {o!r}")


def assert_greyscale_distinguishable(note: str = ""):
    """Not an automated pixel check -- a documented manual step. Every
    script in this set uses a DIFFERENT linestyle AND a DIFFERENT marker
    per series (see LINESTYLES/MARKERS above) specifically so that colour
    is never the only channel carrying identity; rendering a figure at
    grayscale (e.g. via a PDF colour-blind simulator) should keep every
    series visually distinct by dash pattern/marker shape alone."""
    pass
