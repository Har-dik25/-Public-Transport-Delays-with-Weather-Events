"""
utils.py — Utility / Helper Functions
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

REPORTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reports")
FIGURES_DIR = os.path.join(REPORTS_DIR, "figures")


def save_figure(fig, name, dpi=150):
    """Save a matplotlib figure to reports/figures/."""
    os.makedirs(FIGURES_DIR, exist_ok=True)
    filepath = os.path.join(FIGURES_DIR, f"{name}.png")
    fig.savefig(filepath, dpi=dpi, bbox_inches="tight", facecolor="white")
    print(f"   📊 Saved figure → {filepath}")
    return filepath


def set_plot_style():
    """Set a consistent, professional plotting style."""
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({
        "figure.figsize": (12, 6),
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.dpi": 100,
    })


def print_section(title):
    """Print a formatted section header."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def data_quality_report(df, name="Dataset"):
    """Print a data quality summary."""
    print_section(f"Data Quality Report: {name}")
    print(f"Shape: {df.shape}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    print(f"\nMissing Values:")
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if len(missing) == 0:
        print("  None! ✅")
    else:
        for col, count in missing.items():
            print(f"  {col}: {count} ({count/len(df)*100:.1f}%)")
    print(f"\nDuplicates: {df.duplicated().sum()}")
    print(f"\nColumn Types:")
    for dtype, count in df.dtypes.value_counts().items():
        print(f"  {dtype}: {count}")
