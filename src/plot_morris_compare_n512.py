# LEGACY / 旧版：不属于当前 FSIM 分析入口。
# Status: legacy_result_processing; see config/project_registry.json and config/README.md.
# Retained for historical traceability; current results require separate validation.
import os

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch


PARAM_ABBR = {
    "dispersal__rain_min": "DIRM",
    "infection__infection_condition": "IC",
    "zoospore_alive__vpd": "ZVPD",
    "mp_tmp__rh_wet_night": "MTWN",
    "sporulation__Tmax": "STMX",
    "dispersal__wind_min": "DIWM",
    "sporulation__Topt": "STOP",
    "sporulation__MP_min": "SMMI",
    "deposition__rain_min": "DERM",
    "sporulation__Tmin": "STMI",
    "incubation__spl_end": "ISED",
    "zoospore_alive__rain": "ZARA",
    "spore__alpha_pos": "SAP",
    "spore__alpha_neg": "SAN",
}
SPORE_PARAMS = {"spore__alpha_pos", "spore__alpha_neg"}


def _read(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return pd.read_excel(path)


def _abbr(series: pd.Series) -> pd.Series:
    return series.map(PARAM_ABBR).fillna(series)


def _plot_mae_panel(ax, df: pd.DataFrame, show_ylabels: bool, highlight_spore: bool) -> None:
    d = df.sort_values("mu_star", ascending=False).copy()
    y = range(len(d))
    colors = ["#4C78A8"] * len(d)
    if highlight_spore:
        for i, p in enumerate(d["Parameter"].tolist()):
            if p in {"spore__alpha_pos", "spore__alpha_neg"}:
                colors[i] = "#F28E2B"
    xerr = d["mu_star_std"] if "mu_star_std" in d.columns else None
    ax.barh(y, d["mu_star"], color=colors, alpha=0.85, xerr=xerr, ecolor="#555555", capsize=2)
    ax.set_yticks(list(y))
    ax.set_yticklabels(_abbr(d["Parameter"]) if show_ylabels else [])
    ax.set_xlabel(r"$\mu^\ast$", fontsize=12)
    ax.tick_params(axis="both", direction="in")
    ax.grid(axis="x", color="#DDDDDD", linewidth=0.6, linestyle="--")
    ax.invert_yaxis()


def _plot_signed_panel(ax, df: pd.DataFrame, show_ylabels: bool, highlight_spore: bool = False) -> None:
    d = df.sort_values("mu", ascending=False).copy()
    y = range(len(d))
    colors = []
    for p, v in zip(d["Parameter"].tolist(), d["mu"].tolist()):
        if highlight_spore and p in SPORE_PARAMS:
            colors.append("#E15759")  # spore parameters
        else:
            colors.append("#4C78A8" if v < 0 else "#F28E2B")  # common parameters
    xerr = d["mu_std"] if "mu_std" in d.columns else None
    ax.barh(y, d["mu"], color=colors, xerr=xerr, ecolor="#555555", capsize=2)
    ax.axvline(0, color="0.25", lw=0.8)
    ax.set_yticks(list(y))
    ax.set_yticklabels(_abbr(d["Parameter"]) if show_ylabels else [])
    ax.set_xlabel(r"$\mu$", fontsize=12)
    ax.tick_params(axis="both", direction="in")
    ax.grid(axis="x", color="#DDDDDD", linewidth=0.6, linestyle="--")
    ax.invert_yaxis()


def main() -> None:
    src_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(src_dir, ".."))
    sum_dir = os.path.join(project_root, "result", "morris_sum")
    fig_dir = os.path.join(project_root, "fig")
    os.makedirs(fig_dir, exist_ok=True)

    base_mae = _read(os.path.join(sum_dir, "morris_baseline_MAE_N512_agg.xlsx"))
    base_sig = _read(os.path.join(sum_dir, "morris_baseline_Signed_N512_agg.xlsx"))
    spore_mae = _read(os.path.join(sum_dir, "morris_spore_infthr_v2_MAE_N512_agg.xlsx"))
    spore_sig = _read(os.path.join(sum_dir, "morris_spore_infthr_v2_Signed_N512_agg.xlsx"))

    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 12,
            "mathtext.fontset": "custom",
            "mathtext.rm": "Times New Roman",
            "mathtext.it": "Times New Roman:italic",
            "mathtext.bf": "Times New Roman:bold",
        }
    )

    # Figure 1: MAE, 1 row x 2 cols (double-column width, each subplot ~single-column)
    fig1, axes1 = plt.subplots(1, 2, figsize=(7.08, 5.4))
    _plot_mae_panel(axes1[0], base_mae, show_ylabels=True, highlight_spore=False)
    _plot_mae_panel(axes1[1], spore_mae, show_ylabels=True, highlight_spore=True)
    base_mae_top = (
        base_mae["mu_star"] + base_mae.get("mu_star_std", pd.Series(0.0, index=base_mae.index))
    ).max()
    spore_mae_top = (
        spore_mae["mu_star"] + spore_mae.get("mu_star_std", pd.Series(0.0, index=spore_mae.index))
    ).max()
    mae_max = max(float(base_mae_top), float(spore_mae_top))
    for ax in axes1:
        ax.set_xlim(0, mae_max * 1.05)
    legend_handles = [
        Patch(facecolor="#4C78A8", alpha=0.85, label=r"Common parameters ($\mu^\ast$)"),
        Patch(facecolor="#F28E2B", alpha=0.85, label=r"Spore parameters ($\mu^\ast$)"),
    ]
    fig1.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.55, 0.975),
        ncol=2,
        frameon=False,
        fontsize=12,
    )
    axes1[0].text(0.98, 0.03, "(a)", transform=axes1[0].transAxes, ha="right", va="bottom")
    axes1[1].text(0.98, 0.03, "(b)", transform=axes1[1].transAxes, ha="right", va="bottom")
    fig1.tight_layout(rect=[0, 0, 1, 0.94])
    out1 = os.path.join(fig_dir, "morris_compare_MAE_panels_N512.png")
    fig1.savefig(out1, dpi=300)
    plt.show()

    # Figure 2: Signed, 1 row x 2 cols
    fig2, axes2 = plt.subplots(1, 2, figsize=(7.08, 5.4))
    _plot_signed_panel(axes2[0], base_sig, show_ylabels=True, highlight_spore=False)
    _plot_signed_panel(axes2[1], spore_sig, show_ylabels=True, highlight_spore=True)
    base_sig_lo = (
        base_sig["mu"] - base_sig.get("mu_std", pd.Series(0.0, index=base_sig.index))
    ).min()
    base_sig_hi = (
        base_sig["mu"] + base_sig.get("mu_std", pd.Series(0.0, index=base_sig.index))
    ).max()
    spore_sig_lo = (
        spore_sig["mu"] - spore_sig.get("mu_std", pd.Series(0.0, index=spore_sig.index))
    ).min()
    spore_sig_hi = (
        spore_sig["mu"] + spore_sig.get("mu_std", pd.Series(0.0, index=spore_sig.index))
    ).max()
    sig_min = min(float(base_sig_lo), float(spore_sig_lo))
    sig_max = max(float(base_sig_hi), float(spore_sig_hi))
    pad = (sig_max - sig_min) * 0.05 if sig_max > sig_min else 0.1
    for ax in axes2:
        ax.set_xlim(sig_min - pad, sig_max + pad)
    legend_handles2 = [
        Patch(facecolor="#4C78A8", label=r"Common parameters ($\mu<0$)"),
        Patch(facecolor="#F28E2B", label=r"Common parameters ($\mu\geq 0$)"),
        Patch(facecolor="#E15759", label=r"Spore parameters ($\mu$)"),
    ]
    fig2.legend(
        handles=legend_handles2,
        loc="upper center",
        bbox_to_anchor=(0.50, 0.965),
        ncol=3,
        frameon=False,
        fontsize=12,
        columnspacing=0.6,
        handletextpad=0.4,
    )
    axes2[0].text(0.98, 0.03, "(a)", transform=axes2[0].transAxes, ha="right", va="bottom")
    axes2[1].text(0.98, 0.03, "(b)", transform=axes2[1].transAxes, ha="right", va="bottom")
    fig2.tight_layout(rect=[0, 0, 1, 0.94])
    out2 = os.path.join(fig_dir, "morris_compare_Signed_panels_N512.png")
    fig2.savefig(out2, dpi=300)
    plt.show()


if __name__ == "__main__":
    main()
