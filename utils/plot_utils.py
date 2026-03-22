"""
utils/plot_utils.py
--------------------
Standardized charting utilities for AAM research.
All charts use the AAM color palette (navy / gold / blue)
and consistent styling for professional presentation.
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
import numpy as np
from utils.config import COLORS


def set_aam_style():
    """Apply AAM house style to all matplotlib charts."""
    plt.rcParams.update({
        "figure.facecolor":     "white",
        "axes.facecolor":       "white",
        "axes.edgecolor":       "#CCCCCC",
        "axes.labelcolor":      COLORS["navy"],
        "axes.titlecolor":      COLORS["navy"],
        "axes.titleweight":     "bold",
        "axes.titlesize":       13,
        "axes.labelsize":       10,
        "axes.grid":            True,
        "grid.color":           "#EEEEEE",
        "grid.linewidth":       0.8,
        "xtick.color":          COLORS["gray"],
        "ytick.color":          COLORS["gray"],
        "legend.fontsize":      9,
        "legend.framealpha":    0.9,
        "font.family":          "sans-serif",
        "figure.dpi":           150,
    })


def waterfall_chart(
    values: list,
    labels: list,
    title: str = "Waterfall",
    figsize: tuple = (12, 6),
    save_path: str = None,
) -> plt.Figure:
    """
    Build a waterfall chart — useful for EV bridge,
    FCF build-up, or returns attribution.

    Parameters
    ----------
    values : list of float
        Values for each bar (positive = up, negative = down)
    labels : list of str
        Labels for each bar
    """
    set_aam_style()
    fig, ax = plt.subplots(figsize=figsize)

    running = 0
    bottoms, tops, colors = [], [], []
    for v in values:
        bottoms.append(min(running, running + v))
        tops.append(abs(v))
        colors.append(COLORS["green"] if v >= 0 else COLORS["red"])
        running += v

    bars = ax.bar(labels, tops, bottom=bottoms, color=colors,
                  alpha=0.85, width=0.6, edgecolor="white", linewidth=0.5)

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_y() + bar.get_height() + abs(max(values)) * 0.01,
            f"${val:+,.0f}M" if abs(val) >= 1 else f"${val:+.1f}M",
            ha="center", va="bottom", fontsize=9, color=COLORS["navy"],
            fontweight="bold"
        )

    ax.set_title(title, pad=12)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"${x:,.0f}M"))
    ax.axhline(0, color=COLORS["gray"], linewidth=0.8)
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def revenue_chart(
    years: list,
    revenues: list,
    title: str = "Revenue ($M)",
    bar_color: str = None,
    figsize: tuple = (10, 5),
    save_path: str = None,
) -> plt.Figure:
    """Bar chart of annual revenue with YoY growth labels."""
    set_aam_style()
    bar_color = bar_color or COLORS["navy"]
    fig, ax = plt.subplots(figsize=figsize)

    bars = ax.bar(years, revenues, color=bar_color, alpha=0.85,
                  width=0.6, edgecolor="white")

    for i, (bar, rev) in enumerate(zip(bars, revenues)):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(revenues) * 0.01,
                f"${rev:,.0f}M", ha="center", va="bottom",
                fontsize=9, color=COLORS["navy"])
        if i > 0:
            growth = (revenues[i] / revenues[i-1] - 1) * 100
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() / 2,
                    f"+{growth:.0f}%", ha="center", va="center",
                    fontsize=8, color="white", fontweight="bold")

    ax.set_title(title, pad=12)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"${x:,.0f}M"))
    ax.set_ylim(0, max(revenues) * 1.15)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def margin_trend_chart(
    periods: list,
    gross_margins: list = None,
    ebitda_margins: list = None,
    net_margins: list = None,
    title: str = "Margin Trends",
    figsize: tuple = (10, 5),
    save_path: str = None,
) -> plt.Figure:
    """Line chart of margin trends across periods."""
    set_aam_style()
    fig, ax = plt.subplots(figsize=figsize)

    if gross_margins:
        ax.plot(periods, gross_margins, marker="o", linewidth=2,
                color=COLORS["navy"], label="Gross margin")
    if ebitda_margins:
        ax.plot(periods, ebitda_margins, marker="s", linewidth=2,
                color=COLORS["gold"], label="EBITDA margin")
    if net_margins:
        ax.plot(periods, net_margins, marker="^", linewidth=2,
                color=COLORS["blue"], label="Net margin")

    ax.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"{x:.0f}%"))
    ax.set_title(title, pad=12)
    ax.legend()
    ax.axhline(0, color=COLORS["gray"], linewidth=0.5)
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def comps_scatter(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    label_col: str = "ticker",
    highlight: str = None,
    title: str = "Comps scatter",
    figsize: tuple = (10, 6),
    save_path: str = None,
) -> plt.Figure:
    """
    Scatter plot for comparable company analysis.
    Highlights the target company vs. peers.
    """
    set_aam_style()
    df = df.dropna(subset=[x_col, y_col])
    fig, ax = plt.subplots(figsize=figsize)

    for _, row in df.iterrows():
        is_target = highlight and row[label_col] == highlight
        ax.scatter(row[x_col], row[y_col],
                   color=COLORS["gold"] if is_target else COLORS["navy"],
                   s=120 if is_target else 80,
                   zorder=5 if is_target else 3,
                   edgecolors="white", linewidth=0.5)
        ax.annotate(row[label_col],
                    (row[x_col], row[y_col]),
                    textcoords="offset points",
                    xytext=(6, 4), fontsize=8,
                    color=COLORS["gold"] if is_target else COLORS["gray"],
                    fontweight="bold" if is_target else "normal")

    ax.set_xlabel(x_col.replace("_", " ").title())
    ax.set_ylabel(y_col.replace("_", " ").title())
    ax.set_title(title, pad=12)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
