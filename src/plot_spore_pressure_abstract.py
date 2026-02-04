from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = PROJECT_ROOT / "fig"


def main() -> None:
    # Schematic pressure curve: negative first, positive later, two peaks each
    x = np.linspace(0, 1, 500)
    y = np.zeros_like(x)
    # Negative half with two troughs
    neg = x <= 0.5
    y[neg] = -0.45 * np.sin(2 * np.pi * x[neg]) ** 2 - 0.08
    # Positive half with two peaks
    pos = x > 0.5
    y[pos] = 0.45 * np.sin(2 * np.pi * (x[pos] - 0.5)) ** 2 + 0.08

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 2.6))
    ax.plot(x, y, color="#2B6CB0", lw=1.8)
    ax.fill_between(x, 0, y, where=y >= 0, color="#90CDF4", alpha=0.6)
    ax.axhline(0.0, color="0.7", lw=0.8)

    # Remove axes for graphical abstract
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.tight_layout()
    out_path = FIG_DIR / "spore_pressure_abstract_schematic.png"
    fig.savefig(out_path, dpi=300)
    print(f"Saved: {out_path}")

    # Unsmooth schematic: all positive with 2–3 small peaks and 1–2 large peaks
    x2 = np.linspace(0, 1, 600)
    y2 = (
        0.12 * np.exp(-((x2 - 0.15) / 0.04) ** 2) +
        0.18 * np.exp(-((x2 - 0.28) / 0.05) ** 2) +
        0.45 * np.exp(-((x2 - 0.55) / 0.06) ** 2) +
        0.10 * np.exp(-((x2 - 0.72) / 0.04) ** 2) +
        0.35 * np.exp(-((x2 - 0.88) / 0.05) ** 2)
    )

    fig2, ax2 = plt.subplots(figsize=(7, 2.6))
    ax2.plot(x2, y2, color="#2B6CB0", lw=1.8)
    ax2.fill_between(x2, 0, y2, color="#90CDF4", alpha=0.6)
    ax2.axhline(0.0, color="0.7", lw=0.8)
    ax2.set_xticks([])
    ax2.set_yticks([])
    for spine in ax2.spines.values():
        spine.set_visible(False)
    fig2.tight_layout()
    out_path2 = FIG_DIR / "spore_pressure_abstract_unsmoothed.png"
    fig2.savefig(out_path2, dpi=300)
    print(f"Saved: {out_path2}")


if __name__ == "__main__":
    main()
