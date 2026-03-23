from pathlib import Path
import json

import matplotlib.pyplot as plt


def _load_final(calib_sum_dir: Path) -> tuple[dict, dict]:
    b_path = calib_sum_dir / "final_baseline_params.json"
    s_path = calib_sum_dir / "final_spore_params.json"
    if not b_path.exists():
        raise FileNotFoundError(f"Missing file: {b_path}")
    if not s_path.exists():
        raise FileNotFoundError(f"Missing file: {s_path}")
    with open(b_path, "r", encoding="utf-8") as f:
        b = json.load(f)
    with open(s_path, "r", encoding="utf-8") as f:
        s = json.load(f)
    return b, s


def _plot_panel(
    ax,
    baseline_row: dict,
    spore_row: dict,
    prefix: str,
    panel_tag: str,
) -> None:
    def _pick(d: dict, keys: list[str], default: float = 0.0) -> float:
        for k in keys:
            if k in d and d[k] is not None:
                try:
                    return float(d[k])
                except Exception:
                    continue
        return float(default)

    # Grouped bars: baseline vs spore, each with MAE and RMSE
    x = [0.0, 1.0]
    width = 0.288
    mae_vals = [
        _pick(baseline_row, [f"{prefix}_mae_mean", f"{prefix}_mae"]),
        _pick(spore_row, [f"{prefix}_mae_mean", f"{prefix}_mae"]),
    ]
    rmse_vals = [
        _pick(baseline_row, [f"{prefix}_rmse_mean", f"{prefix}_rmse"]),
        _pick(spore_row, [f"{prefix}_rmse_mean", f"{prefix}_rmse"]),
    ]
    mae_err = [
        _pick(baseline_row, [f"{prefix}_mae_std"], 0.0),
        _pick(spore_row, [f"{prefix}_mae_std"], 0.0),
    ]
    rmse_err = [
        _pick(baseline_row, [f"{prefix}_rmse_std"], 0.0),
        _pick(spore_row, [f"{prefix}_rmse_std"], 0.0),
    ]

    bars_mae = ax.bar(
        [xi - width / 2 for xi in x],
        mae_vals,
        yerr=mae_err,
        width=width,
        color="#4C78A8",
        ecolor="#666666",
        edgecolor="black",
        linewidth=0.6,
        capsize=3,
        alpha=0.9,
        label="MAE",
    )
    bars_rmse = ax.bar(
        [xi + width / 2 for xi in x],
        rmse_vals,
        yerr=rmse_err,
        width=width,
        color="#F28E2B",
        ecolor="#666666",
        edgecolor="black",
        linewidth=0.6,
        capsize=3,
        alpha=0.9,
        label="RMSE",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(["FSIM", "FSIM-S"])
    ax.tick_params(axis="both", direction="in")
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.6, linestyle="--")
    ax.text(0.05, 0.96, panel_tag, transform=ax.transAxes, ha="left", va="top")
    # value labels in bar center
    all_bars = list(bars_mae) + list(bars_rmse)
    for b in all_bars:
        h = b.get_height()

        # 更稳健的“柱内居中”（直接用 bar 的 bbox）
        y_pos = b.get_y() + h / 2

        ax.text(
            b.get_x() + b.get_width() / 2,
            y_pos,
            f"{h:.1f}",   # ✅ 保留1位小数（自动四舍五入）
            ha="center",
            va="center",
            fontsize=12,
            fontfamily="Times New Roman",
        )

    # Reduction labels (%) above spore bars (relative to baseline)
    b_mae = mae_vals[0]
    s_mae = mae_vals[1]
    b_rmse = rmse_vals[0]
    s_rmse = rmse_vals[1]
    mae_pct = ((b_mae - s_mae) / b_mae * 100.0) if b_mae != 0 else 0.0
    rmse_pct = ((b_rmse - s_rmse) / b_rmse * 100.0) if b_rmse != 0 else 0.0
    spore_mae_bar = bars_mae[1]
    spore_rmse_bar = bars_rmse[1]
    y_reduct_mae = (s_mae + mae_err[1] + 0.05) * 1.08 + (1.0 if prefix == "train" else 0.5)
    ax.text(
        spore_mae_bar.get_x() + spore_mae_bar.get_width() / 2,
        y_reduct_mae,
        f"\u2193 {mae_pct:.1f}%",
        ha="center",
        va="bottom",
        fontsize=12,
        fontfamily="Times New Roman",
    )
    y_reduct_rmse = (s_rmse + rmse_err[1] + 0.05) * 1.05 + 0.5
    ax.text(
        spore_rmse_bar.get_x() + spore_rmse_bar.get_width() / 2,
        y_reduct_rmse,
        f"\u2193 {rmse_pct:.1f}%",
        ha="center",
        va="bottom",
        fontsize=12,
        fontfamily="Times New Roman",
    )

def main() -> None:
    src_dir = Path(__file__).resolve().parent
    project_root = src_dir.parent
    calib_sum_dir = project_root / "result" / "calibration_sum"
    b_json, s_json = _load_final(calib_sum_dir)

    plt.rcParams.update({"font.family": "Times New Roman", "font.size": 12})
    b_row = b_json.get("metrics_seed_summary", {})
    s_row = s_json.get("metrics_seed_summary", {})
    if not b_row or not s_row:
        raise ValueError("Missing 'metrics_seed_summary' in final parameter json files.")
    seed_b = b_json.get("selected_seed")
    seed_s = s_json.get("selected_seed")
    if seed_b != seed_s:
        raise ValueError(f"Seed mismatch between final json files: baseline={seed_b}, spore={seed_s}")

    fig, axes = plt.subplots(1, 2, figsize=(7.08, 3.2))
    _plot_panel(axes[0], b_row, s_row, "train", "(a)")
    _plot_panel(axes[1], b_row, s_row, "val", "(b)")
    axes[1].legend(frameon=False, fontsize=10, loc="upper right")

    for ax in axes:
        ax.set_ylim(0, 17.5)

    axes[0].set_ylabel("Error (days)")
    fig.tight_layout()
    out = project_root / "fig" / "bayes_core_bar_1x2.png"
    fig.savefig(out, dpi=300)
    plt.show()


if __name__ == "__main__":
    main()
