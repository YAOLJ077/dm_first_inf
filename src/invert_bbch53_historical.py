#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    def tqdm(x, **kwargs):  # type: ignore
        return x

SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    from src.bbch53_module import load_hourly_weather_from_json, compute_agdd_hourly
except Exception:  # pragma: no cover
    from bbch53_module import load_hourly_weather_from_json, compute_agdd_hourly


VARIETY_CODE_MAP = {"wk": "WK", "jf": "JF", "mpt": "MPT"}


def _parse_grid_index_file(index_path: Path) -> pd.DataFrame:
    # Preferred path: structured table with header columns.
    for sep in [",", "\t", None]:
        try:
            df = pd.read_csv(index_path, sep=sep, engine="python")
            cols = {str(c).strip().lower(): c for c in df.columns}
            required = ["year", "grid_id", "latitude", "longitude", "json_file_path"]
            if all(k in cols for k in required):
                out = pd.DataFrame(
                    {
                        "year": pd.to_numeric(df[cols["year"]], errors="coerce").astype("Int64"),
                        "grid_id": pd.to_numeric(df[cols["grid_id"]], errors="coerce").astype("Int64"),
                        "latitude": pd.to_numeric(df[cols["latitude"]], errors="coerce"),
                        "longitude": pd.to_numeric(df[cols["longitude"]], errors="coerce"),
                        "json_rel_path": df[cols["json_file_path"]].astype(str).str.strip(),
                    }
                )
                out = out.dropna(subset=["year", "grid_id", "latitude", "longitude", "json_rel_path"]).copy()
                if not out.empty:
                    out["year"] = out["year"].astype(int)
                    out["grid_id"] = out["grid_id"].astype(int)
                    return out.reset_index(drop=True)
        except Exception:
            pass

    rows: List[Dict] = []
    raw_lines = index_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    for line in raw_lines:
        s = line.strip()
        if (not s) or s.startswith("#"):
            continue
        s = (
            s.replace("，", ",")
            .replace("（", " ")
            .replace("）", " ")
            .replace("：", ":")
        )
        # skip header-like lines
        low = s.lower()
        if ("year" in low and "grid" in low and "json" in low) or ("grid_id" in low):
            continue

        # Robust pattern: year, grid_id, lat, lon, json_path(.json)
        m = re.search(
            r"(?P<year>\d{4})\s*[, \t]+\s*(?P<grid>-?\d+)\s*[, \t]+\s*"
            r"(?P<lat>-?\d+(?:\.\d+)?)\s*[, \t]+\s*"
            r"(?P<lon>-?\d+(?:\.\d+)?)\s*[, \t]+\s*"
            r"(?P<path>\S+?\.json)",
            s,
            flags=re.IGNORECASE,
        )
        if not m:
            parts = re.split(r"[,\t ]+", s)
            if len(parts) >= 5:
                try:
                    year = int(parts[0])
                    grid_id = int(parts[1])
                    lat = float(parts[2])
                    lon = float(parts[3])
                    json_rel = parts[4]
                except Exception:
                    continue
            else:
                continue
        else:
            try:
                year = int(m.group("year"))
                grid_id = int(m.group("grid"))
                lat = float(m.group("lat"))
                lon = float(m.group("lon"))
                json_rel = m.group("path")
            except Exception:
                continue

        json_rel = json_rel.strip().strip('"').strip("'")
        rows.append(
            {
                "year": year,
                "grid_id": grid_id,
                "latitude": lat,
                "longitude": lon,
                "json_rel_path": json_rel,
            }
        )
    if not rows:
        preview = "\n".join(raw_lines[:8])
        raise ValueError(
            f"No valid rows parsed from index file: {index_path}\n"
            f"First lines preview:\n{preview}"
        )
    return pd.DataFrame(rows)


def _find_index_file(variety_dir: Path) -> Path:
    cands = [p for p in variety_dir.iterdir() if p.is_file() and p.name.lower().startswith("grid_index")]
    if not cands:
        raise FileNotFoundError(f"No grid_index file found under: {variety_dir}")
    cands = sorted(cands, key=lambda p: p.name)
    return cands[0]


def _start_dt_for_year_from_doy(year: int, doy: int) -> pd.Timestamp:
    doy_eff = int(np.clip(int(doy), 1, 366))
    return pd.Timestamp(year=int(year), month=1, day=1) + pd.Timedelta(days=doy_eff - 1)


def _estimate_date_with_gate(
    hourly_df: pd.DataFrame,
    theta_agdd: float,
    t_base: float,
    start_dt: pd.Timestamp,
    use_rgdd_gate: bool,
    rgdd_min: float,
    rgdd_window: int,
) -> pd.Timestamp:
    ag = compute_agdd_hourly(hourly_df, t_base=t_base, start_datetime=start_dt)
    if ag.empty:
        return pd.NaT
    d = (
        ag.assign(date=ag["datetime"].dt.normalize())
        .groupby("date", as_index=False)
        .agg(gdd_day=("gdd_hour", "sum"), agdd_end=("agdd", "max"))
        .sort_values("date")
    )
    if use_rgdd_gate:
        d["rgdd_window"] = d["gdd_day"].rolling(int(rgdd_window), min_periods=int(rgdd_window)).sum()
        mask = (d["agdd_end"] >= float(theta_agdd)) & (d["rgdd_window"] >= float(rgdd_min))
    else:
        mask = d["agdd_end"] >= float(theta_agdd)
    hit = d[mask]
    if hit.empty:
        return pd.NaT
    return pd.to_datetime(hit.iloc[0]["date"]).normalize()


def _load_hourly_year(json_path: Path, year: int) -> pd.DataFrame:
    df = load_hourly_weather_from_json(json_path)
    return df[df["datetime"].dt.year == int(year)].reset_index(drop=True)


def _invert_one_task(task: Dict) -> Dict:
    json_path = Path(task["json_path"])
    year = int(task["year"])
    vcode = str(task["variety"])
    hourly = _load_hourly_year(json_path, year)
    pred = pd.NaT
    if not hourly.empty:
        year_term = float(task["k_mpt"]) * float(task["year_signal"]) if vcode == "MPT" else 0.0
        scale = float(np.clip(1.0 + float(task["delta"]) + year_term, 0.2, 3.0))
        theta_eff = float(task["theta_agdd"]) * scale
        start_doy_eff = int(np.clip(int(task["start_doy"]) + int(task["start_shift"]), 1, 366))
        start_dt = _start_dt_for_year_from_doy(year, start_doy_eff)
        pred = _estimate_date_with_gate(
            hourly_df=hourly,
            theta_agdd=theta_eff,
            t_base=float(task["t_base"]),
            start_dt=start_dt,
            use_rgdd_gate=bool(task["use_rgdd_gate"]),
            rgdd_min=float(task["rgdd_min"]),
            rgdd_window=int(task["rgdd_window"]),
        )

    return {
        "variety": vcode,
        "variety_dir_key": str(task["variety_dir_key"]),
        "year": year,
        "grid_id": int(task["grid_id"]),
        "latitude": float(task["latitude"]),
        "longitude": float(task["longitude"]),
        "bbch53_date": pred.strftime("%Y-%m-%d") if pd.notna(pred) else "",
        "bbch53_doy": int(pred.dayofyear) if pd.notna(pred) else np.nan,
        "json_file_path": str(task["json_rel_path"]),
        "json_abs_path": str(task["json_path"]),
    }


def _mean_temp_janfeb_from_json(json_path: Path, year: int) -> float:
    data = json.loads(json_path.read_text(encoding="utf-8", errors="ignore"))
    rows = data.get("weather_hourly", []) if isinstance(data, dict) else data
    sum_jf = 0.0
    n_jf = 0
    sum_all = 0.0
    n_all = 0
    y_prefix = f"{int(year):04d}-"
    for r in rows:
        dt = str(r.get("datetime", ""))
        if not dt.startswith(y_prefix):
            continue
        try:
            t = float(r.get("air_temp"))
        except Exception:
            continue
        n_all += 1
        sum_all += t
        m = dt[5:7] if len(dt) >= 7 else ""
        if m in ("01", "02"):
            n_jf += 1
            sum_jf += t
    if n_jf > 0:
        return float(sum_jf / n_jf)
    if n_all > 0:
        return float(sum_all / n_all)
    return float("nan")


def _calc_year_signal_for_one_grid(args: Tuple[int, pd.DataFrame, Path]) -> Tuple[int, Dict[int, float]]:
    grid_id, sub, variety_dir = args
    vals: Dict[int, float] = {}
    for r in sub.itertuples(index=False):
        json_path = variety_dir / str(r.json_rel_path)
        if not json_path.exists():
            continue
        vals[int(r.year)] = _mean_temp_janfeb_from_json(json_path, int(r.year))
    if not vals:
        return int(grid_id), {}
    years = sorted(vals.keys())
    arr = np.asarray([vals[y] for y in years], dtype=float)
    mu = float(np.nanmean(arr)) if len(arr) > 0 else 0.0
    sd = float(np.nanstd(arr)) if len(arr) > 0 else 1.0
    if (not np.isfinite(sd)) or sd < 1e-8:
        sd = 1.0
    zmap = {int(y): float((float(vals[y]) - mu) / sd) if np.isfinite(vals[y]) else 0.0 for y in years}
    return int(grid_id), zmap


def _calc_year_signal_by_grid(df_index: pd.DataFrame, variety_dir: Path, max_workers: int = 32) -> Dict[Tuple[int, int], float]:
    out: Dict[Tuple[int, int], float] = {}
    groups = list(df_index.groupby("grid_id"))
    if not groups:
        return out
    workers = max(1, min(int(max_workers), 64))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_calc_year_signal_for_one_grid, (int(gid), sub.copy(), variety_dir)) for gid, sub in groups]
        for fut in tqdm(as_completed(futs), total=len(futs), desc=f"{variety_dir.name.upper()} year-signal"):
            grid_id, zmap = fut.result()
            for year, z in zmap.items():
                out[(int(year), int(grid_id))] = float(z)
    return out


def _resolve_weather_path(variety_dir: Path, year: int, grid_id: int, df_index: pd.DataFrame) -> Optional[Path]:
    row = df_index[(df_index["year"] == int(year)) & (df_index["grid_id"] == int(grid_id))]
    if row.empty:
        return None
    json_rel = str(row.iloc[0]["json_rel_path"])
    p = variety_dir / json_rel
    return p if p.exists() else None


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Invert historical BBCH53 dates for wk/jf/mpt on all grid-year weather JSON files.")
    p.add_argument("--input-root", type=str, default="../input_weather_json", help="Relative to project root.")
    p.add_argument("--output-root", type=str, default="../bbch53_date", help="Relative to project root.")
    p.add_argument("--varieties", type=str, default="wk,jf,mpt")
    p.add_argument("--year-min", type=int, default=2000)
    p.add_argument("--year-max", type=int, default=2024)
    p.add_argument("--use-rgdd-gate", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--rgdd-window", type=int, default=7)
    p.add_argument("--t-base", type=float, default=10.0)
    p.add_argument("--theta-agdd", type=float, default=280.0)
    p.add_argument("--start-doy", type=int, default=53)
    p.add_argument("--rgdd-min", type=float, default=60.0)
    p.add_argument("--delta-wk", type=float, default=-0.05)
    p.add_argument("--delta-jf", type=float, default=0.05)
    p.add_argument("--delta-mpt", type=float, default=0.95)
    p.add_argument("--start-shift-wk", type=int, default=6)
    p.add_argument("--start-shift-jf", type=int, default=0)
    p.add_argument("--start-shift-mpt", type=int, default=-14)
    p.add_argument("--k-mpt", type=float, default=-0.7)
    p.add_argument(
        "--export-csrb-json",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If true, write extra JSON files with BBCH_53 injected for direct scheduler use.",
    )
    p.add_argument(
        "--export-result-json",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If true, write lightweight result JSON per variety/year/grid.",
    )
    p.add_argument("--n-jobs", type=int, default=128, help="Number of worker processes.")
    p.add_argument("--chunksize", type=int, default=16, help="Task chunksize for process pool.")
    p.add_argument(
        "--year-signal-scope",
        type=str,
        default="mpt_only",
        choices=["mpt_only", "all", "none"],
        help="Which varieties to compute year_signal for.",
    )
    p.add_argument("--year-signal-workers", type=int, default=32, help="Workers for year_signal preprocessing.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    input_root = (PROJECT_ROOT / args.input_root).resolve()
    output_root = (PROJECT_ROOT / args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    var_list = [v.strip().lower() for v in str(args.varieties).split(",") if v.strip()]
    dmap = {"WK": float(args.delta_wk), "JF": float(args.delta_jf), "MPT": float(args.delta_mpt)}
    smap = {"WK": int(args.start_shift_wk), "JF": int(args.start_shift_jf), "MPT": int(args.start_shift_mpt)}
    kmpt = float(args.k_mpt)

    run_cfg = {
        "global_params": {
            "t_base": float(args.t_base),
            "theta_agdd": float(args.theta_agdd),
            "start_doy": int(args.start_doy),
            "use_rgdd_gate": bool(args.use_rgdd_gate),
            "rgdd_min": float(args.rgdd_min),
            "rgdd_window": int(args.rgdd_window),
        },
        "variety_params": {
            "WK": {"delta": dmap["WK"], "start_shift": smap["WK"], "k_mpt": 0.0},
            "JF": {"delta": dmap["JF"], "start_shift": smap["JF"], "k_mpt": 0.0},
            "MPT": {"delta": dmap["MPT"], "start_shift": smap["MPT"], "k_mpt": kmpt},
        },
        "years": {"min": int(args.year_min), "max": int(args.year_max)},
        "input_root": str(input_root),
        "output_root": str(output_root),
    }
    (output_root / "bbch53_inversion_config.json").write_text(json.dumps(run_cfg, indent=2), encoding="utf-8")

    tasks: List[Dict] = []

    for v in tqdm(var_list, total=len(var_list), desc="Prepare varieties"):
        if v not in VARIETY_CODE_MAP:
            print(f"[WARN] Skip unknown variety: {v}")
            continue
        vcode = VARIETY_CODE_MAP[v]
        vdir = input_root / v
        if not vdir.exists():
            print(f"[WARN] Missing variety directory: {vdir}")
            continue

        idx_path = _find_index_file(vdir)
        df_idx = _parse_grid_index_file(idx_path)
        df_idx = df_idx[(df_idx["year"] >= int(args.year_min)) & (df_idx["year"] <= int(args.year_max))].copy()
        if df_idx.empty:
            print(f"[WARN] No rows in year range for {v}")
            continue

        need_year_signal = False
        if str(args.year_signal_scope) == "all":
            need_year_signal = True
        elif str(args.year_signal_scope) == "mpt_only":
            need_year_signal = (vcode == "MPT")
        elif str(args.year_signal_scope) == "none":
            need_year_signal = False
        if need_year_signal and (vcode == "MPT") and abs(float(kmpt)) > 1e-12:
            year_signal_map = _calc_year_signal_by_grid(
                df_index=df_idx,
                variety_dir=vdir,
                max_workers=int(args.year_signal_workers),
            )
        elif need_year_signal and (vcode != "MPT"):
            year_signal_map = _calc_year_signal_by_grid(
                df_index=df_idx,
                variety_dir=vdir,
                max_workers=int(args.year_signal_workers),
            )
        else:
            year_signal_map = {}
        for r in df_idx.itertuples(index=False):
            year = int(r.year)
            grid_id = int(r.grid_id)
            lat = float(r.latitude)
            lon = float(r.longitude)
            json_path = _resolve_weather_path(vdir, year, grid_id, df_idx)
            if json_path is None:
                continue
            tasks.append(
                {
                    "variety": vcode,
                    "variety_dir_key": v,
                    "year": year,
                    "grid_id": grid_id,
                    "latitude": lat,
                    "longitude": lon,
                    "json_rel_path": str(r.json_rel_path),
                    "json_path": str(json_path),
                    "year_signal": float(year_signal_map.get((year, grid_id), 0.0)),
                    "t_base": float(args.t_base),
                    "theta_agdd": float(args.theta_agdd),
                    "start_doy": int(args.start_doy),
                    "use_rgdd_gate": bool(args.use_rgdd_gate),
                    "rgdd_min": float(args.rgdd_min),
                    "rgdd_window": int(args.rgdd_window),
                    "delta": float(dmap[vcode]),
                    "start_shift": int(smap[vcode]),
                    "k_mpt": float(kmpt),
                }
            )

    if not tasks:
        print("[WARN] No tasks generated.")
        return

    max_workers = max(1, min(int(args.n_jobs), int(os.cpu_count() or 1)))
    all_rows: List[Dict] = []
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_invert_one_task, t) for t in tasks]
        pbar = tqdm(total=len(futures), desc="BBCH53 inversion")
        for fut in as_completed(futures):
            all_rows.append(fut.result())
            pbar.update(1)
        pbar.close()

    out_all = pd.DataFrame(all_rows).sort_values(["variety", "year", "grid_id"]).reset_index(drop=True)
    out_all.to_csv(output_root / "bbch53_dates_all_varieties_2000_2024.csv", index=False)
    print(f"Saved: {output_root / 'bbch53_dates_all_varieties_2000_2024.csv'}")

    for v in var_list:
        if v not in VARIETY_CODE_MAP:
            continue
        vcode = VARIETY_CODE_MAP[v]
        out_v = out_all[out_all["variety"] == vcode].copy()
        if out_v.empty:
            continue
        out_v = out_v[
            [
                "variety",
                "year",
                "grid_id",
                "latitude",
                "longitude",
                "bbch53_date",
                "bbch53_doy",
                "json_file_path",
            ]
        ].sort_values(["year", "grid_id"]).reset_index(drop=True)
        out_dir_v = output_root / v
        out_dir_v.mkdir(parents=True, exist_ok=True)
        out_v.to_csv(out_dir_v / "bbch53_dates_all_years.csv", index=False)
        for y, sub in out_v.groupby("year"):
            sub.to_csv(out_dir_v / f"bbch53_dates_{int(y)}.csv", index=False)

    if bool(args.export_csrb_json):
        pbar_json = tqdm(out_all.itertuples(index=False), total=len(out_all), desc="Export CSRB json")
        for r in pbar_json:
            json_path = Path(str(r.json_abs_path))
            if not json_path.exists():
                continue
            raw = json.loads(json_path.read_text(encoding="utf-8"))
            raw["variety_code"] = str(r.variety)
            raw["Place"] = raw.get("Place", f"grid_{int(r.grid_id)}")
            bbch53_value = str(r.bbch53_date) if str(r.bbch53_date) else None
            raw["BBCH_53"] = bbch53_value
            # Backward compatibility for old scripts that still read HHCH_53.
            raw["HHCH_53"] = bbch53_value
            out_json_dir = output_root / str(r.variety).lower() / str(int(r.year))
            out_json_dir.mkdir(parents=True, exist_ok=True)
            out_json_name = json_path.name.replace(".json", "_with_bbch53.json")
            (out_json_dir / out_json_name).write_text(
                json.dumps(raw, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    if bool(args.export_result_json):
        pbar_res_json = tqdm(out_all.itertuples(index=False), total=len(out_all), desc="Export result json")
        for r in pbar_res_json:
            vkey = str(r.variety).lower()
            y = int(r.year)
            out_json_dir = output_root / vkey / str(y)
            out_json_dir.mkdir(parents=True, exist_ok=True)
            out_name = f"grid_{int(r.grid_id)}_bbch53.json"
            payload = {
                "variety": str(r.variety),
                "year": y,
                "grid_id": int(r.grid_id),
                "latitude": float(r.latitude),
                "longitude": float(r.longitude),
                "bbch53_date": str(r.bbch53_date) if str(r.bbch53_date) else None,
                "bbch53_doy": (None if pd.isna(r.bbch53_doy) else int(r.bbch53_doy)),
                "source_json_file": str(r.json_file_path),
                "params": {
                    "t_base": float(args.t_base),
                    "theta_agdd": float(args.theta_agdd),
                    "start_doy": int(args.start_doy),
                    "use_rgdd_gate": bool(args.use_rgdd_gate),
                    "rgdd_min": float(args.rgdd_min),
                    "rgdd_window": int(args.rgdd_window),
                    "delta_wk": float(args.delta_wk),
                    "delta_jf": float(args.delta_jf),
                    "delta_mpt": float(args.delta_mpt),
                    "start_shift_wk": int(args.start_shift_wk),
                    "start_shift_jf": int(args.start_shift_jf),
                    "start_shift_mpt": int(args.start_shift_mpt),
                    "k_mpt": float(args.k_mpt),
                },
            }
            (out_json_dir / out_name).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved: {output_root / 'bbch53_inversion_config.json'}")


if __name__ == "__main__":
    main()
