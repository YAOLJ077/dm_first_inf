#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    def tqdm(x, **kwargs):  # type: ignore
        return x

try:
    import ray
except Exception:  # pragma: no cover
    ray = None

SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    from src.bbch53_module import load_hourly_weather_from_json
    from src.spore_pressure_module import (
        build_daily_features,
        latent_to_pressure,
        simulate_latent_spore_state,
    )
except Exception:  # pragma: no cover
    from bbch53_module import load_hourly_weather_from_json
    from spore_pressure_module import build_daily_features, latent_to_pressure, simulate_latent_spore_state

os.environ.setdefault("RAY_DISABLE_METRICS", "1")
os.environ.setdefault("RAY_DISABLE_USAGE_STATS", "1")
os.environ.setdefault("RAY_METRICS_EXPORT_PORT", "0")
os.environ.setdefault("RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO", "0")

VARIETY_CODE_MAP = {"wk": "WK", "jf": "JF", "mpt": "MPT"}


def _parse_grid_index_file(index_path: Path) -> pd.DataFrame:
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
        low = s.lower()
        if ("year" in low and "grid" in low and "json" in low) or ("grid_id" in low):
            continue
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
            if len(parts) < 5:
                continue
            try:
                year = int(parts[0])
                grid_id = int(parts[1])
                lat = float(parts[2])
                lon = float(parts[3])
                json_rel = parts[4]
            except Exception:
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
        rows.append(
            {
                "year": year,
                "grid_id": grid_id,
                "latitude": lat,
                "longitude": lon,
                "json_rel_path": str(json_rel).strip().strip('"').strip("'"),
            }
        )
    if not rows:
        raise ValueError(f"No valid rows parsed from index file: {index_path}")
    return pd.DataFrame(rows)


def _find_index_file(variety_dir: Path) -> Path:
    cands = [p for p in variety_dir.iterdir() if p.is_file() and p.name.lower().startswith("grid_index")]
    if not cands:
        raise FileNotFoundError(f"No grid_index file found under: {variety_dir}")
    return sorted(cands, key=lambda p: p.name)[0]


def _resolve_weather_path(variety_dir: Path, year: int, grid_id: int, df_index: pd.DataFrame) -> Optional[Path]:
    row = df_index[(df_index["year"] == int(year)) & (df_index["grid_id"] == int(grid_id))]
    if row.empty:
        return None
    rel = str(row.iloc[0]["json_rel_path"])
    p = variety_dir / rel
    return p if p.exists() else None


def _load_params(params_json: Path) -> Dict:
    payload = json.loads(params_json.read_text(encoding="utf-8"))
    if "spore_pressure_params" in payload:
        p = payload["spore_pressure_params"]
    else:
        p = payload
    required = ["rho", "beta_mp", "beta_rain", "beta_wind", "gamma_vpd", "k_mp", "tmin", "topt", "tmax", "roll_window", "baseline_q"]
    miss = [k for k in required if k not in p]
    if miss:
        raise KeyError(f"Missing keys in params json: {miss}")
    return {k: p[k] for k in required}


def _invert_one_task(task: Dict) -> Dict:
    json_path = Path(str(task["json_path"]))
    year = int(task["year"])
    variety = str(task["variety"])
    params = task["params"]

    hourly = load_hourly_weather_from_json(json_path)
    hourly = hourly[hourly["datetime"].dt.year == year].reset_index(drop=True)
    if hourly.empty:
        return {
            "variety": variety,
            "year": year,
            "grid_id": int(task["grid_id"]),
            "latitude": float(task["latitude"]),
            "longitude": float(task["longitude"]),
            "n_days": 0,
            "sday_mean": np.nan,
            "sday_max": np.nan,
            "sday_roll_max": np.nan,
            "json_file_path": str(task["json_rel_path"]),
            "json_abs_path": str(json_path),
            "series_rows": [],
        }

    daily = build_daily_features(hourly)
    latent = simulate_latent_spore_state(daily, params=params)
    pressure = latent_to_pressure(
        latent_daily=latent,
        roll_window=int(round(float(params["roll_window"]))),
        baseline_q=float(params["baseline_q"]),
    )
    series_df = pressure[["date", "S_day", "S_day_roll", "S_day_raw", "A_latent", "source", "loss"]].copy()
    series_df["date"] = pd.to_datetime(series_df["date"]).dt.strftime("%Y-%m-%d")
    series_rows = series_df.to_dict(orient="records")

    return {
        "variety": variety,
        "year": year,
        "grid_id": int(task["grid_id"]),
        "latitude": float(task["latitude"]),
        "longitude": float(task["longitude"]),
        "n_days": int(len(series_df)),
        "sday_mean": float(pd.to_numeric(series_df["S_day"], errors="coerce").mean()),
        "sday_max": float(pd.to_numeric(series_df["S_day"], errors="coerce").max()),
        "sday_roll_max": float(pd.to_numeric(series_df["S_day_roll"], errors="coerce").max()),
        "json_file_path": str(task["json_rel_path"]),
        "json_abs_path": str(json_path),
        "series_rows": series_rows,
    }


if ray is not None:
    @ray.remote
    def _invert_one_task_ray(task: Dict) -> Dict:
        return _invert_one_task(task)


def _export_result_json(row: Dict, output_root: Path, params_payload: Dict) -> int:
    out_dir = output_root / str(row["variety"]).lower() / str(int(row["year"]))
    out_dir.mkdir(parents=True, exist_ok=True)
    out_name = f"grid_{int(row['grid_id'])}_spore_pressure.json"
    payload = {
        "variety": str(row["variety"]),
        "year": int(row["year"]),
        "grid_id": int(row["grid_id"]),
        "latitude": float(row["latitude"]),
        "longitude": float(row["longitude"]),
        "n_days": int(row["n_days"]),
        "summary": {
            "S_day_mean": float(row["sday_mean"]) if np.isfinite(row["sday_mean"]) else None,
            "S_day_max": float(row["sday_max"]) if np.isfinite(row["sday_max"]) else None,
            "S_day_roll_max": float(row["sday_roll_max"]) if np.isfinite(row["sday_roll_max"]) else None,
        },
        "source_json_file": str(row["json_file_path"]),
        "spore_pressure_params": params_payload,
        "spore_day_series": row["series_rows"],
    }
    (out_dir / out_name).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return 1


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Invert historical spore pressure field for wk/jf/mpt on all grid-year weather JSON files.")
    p.add_argument("--input-root", type=str, default="../input_weather_json", help="Relative to project root.")
    p.add_argument("--output-root", type=str, default="../spore_pressure", help="Relative to project root.")
    p.add_argument("--varieties", type=str, default="wk,jf,mpt")
    p.add_argument("--year-min", type=int, default=2000)
    p.add_argument("--year-max", type=int, default=2024)
    p.add_argument(
        "--params-json",
        type=str,
        default="result/historical_inputs/spore_pressure_calibration/A_alpha0.00/spore_pressure_params_for_inversion.json",
        help="Path to calibrated spore params JSON (relative to project root or absolute).",
    )
    p.add_argument("--backend", type=str, default="ray", choices=["ray", "process", "serial"])
    p.add_argument("--n-jobs", type=int, default=128)
    p.add_argument("--chunksize", type=int, default=16)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    input_root = (PROJECT_ROOT / args.input_root).resolve()
    output_root = (PROJECT_ROOT / args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    params_path = Path(args.params_json).expanduser()
    if not params_path.is_absolute():
        params_path = (PROJECT_ROOT / params_path).resolve()
    if not params_path.exists():
        raise FileNotFoundError(f"Params json not found: {params_path}")
    params_payload = _load_params(params_path)

    tasks: List[Dict] = []
    var_list = [v.strip().lower() for v in str(args.varieties).split(",") if v.strip()]
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
            continue
        for r in df_idx.itertuples(index=False):
            year = int(r.year)
            grid_id = int(r.grid_id)
            json_path = _resolve_weather_path(vdir, year, grid_id, df_idx)
            if json_path is None:
                continue
            tasks.append(
                {
                    "variety": vcode,
                    "year": year,
                    "grid_id": grid_id,
                    "latitude": float(r.latitude),
                    "longitude": float(r.longitude),
                    "json_rel_path": str(r.json_rel_path),
                    "json_path": str(json_path),
                    "params": params_payload,
                }
            )

    if not tasks:
        print("[WARN] No tasks generated.")
        return

    all_rows: List[Dict] = []
    backend = str(args.backend).lower()
    if backend == "serial":
        for t in tqdm(tasks, total=len(tasks), desc="Spore pressure inversion"):
            all_rows.append(_invert_one_task(t))
    elif backend == "process":
        max_workers = max(1, min(int(args.n_jobs), int(os.cpu_count() or 1), 61 if os.name == "nt" else 9999))
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            futs = [ex.submit(_invert_one_task, t) for t in tasks]
            for fut in tqdm(as_completed(futs), total=len(futs), desc="Spore pressure inversion"):
                all_rows.append(fut.result())
    else:
        if ray is None:
            print("[WARN] backend=ray but ray not installed, fallback to process.")
            max_workers = max(1, min(int(args.n_jobs), int(os.cpu_count() or 1), 61 if os.name == "nt" else 9999))
            with ProcessPoolExecutor(max_workers=max_workers) as ex:
                futs = [ex.submit(_invert_one_task, t) for t in tasks]
                for fut in tqdm(as_completed(futs), total=len(futs), desc="Spore pressure inversion"):
                    all_rows.append(fut.result())
        else:
            ncpu = max(1, min(int(args.n_jobs), int(os.cpu_count() or 1)))
            ray.init(num_cpus=ncpu, include_dashboard=False, logging_level="ERROR", ignore_reinit_error=True)
            futs = [_invert_one_task_ray.remote(t) for t in tasks]
            pending = list(futs)
            pbar = tqdm(total=len(futs), desc="Spore pressure inversion")
            while pending:
                ready, pending = ray.wait(pending, num_returns=1)
                all_rows.append(ray.get(ready[0]))
                pbar.update(1)
            pbar.close()
            ray.shutdown()

    out_all = pd.DataFrame(all_rows)
    out_all = out_all.sort_values(["variety", "year", "grid_id"]).reset_index(drop=True)
    out_all.to_csv(output_root / "spore_pressure_all_varieties_2000_2024.csv", index=False)

    for v in sorted(out_all["variety"].dropna().astype(str).unique()):
        out_v = out_all[out_all["variety"] == v].copy()
        out_dir_v = output_root / str(v).lower()
        out_dir_v.mkdir(parents=True, exist_ok=True)
        out_v.to_csv(out_dir_v / "spore_pressure_all_years.csv", index=False)
        for y, sub in out_v.groupby("year"):
            sub.to_csv(out_dir_v / f"spore_pressure_{int(y)}.csv", index=False)

    export_tasks = []
    for r in out_all.itertuples(index=False):
        export_tasks.append(
            {
                "variety": str(r.variety),
                "year": int(r.year),
                "grid_id": int(r.grid_id),
                "latitude": float(r.latitude),
                "longitude": float(r.longitude),
                "n_days": int(r.n_days),
                "sday_mean": float(r.sday_mean) if pd.notna(r.sday_mean) else np.nan,
                "sday_max": float(r.sday_max) if pd.notna(r.sday_max) else np.nan,
                "sday_roll_max": float(r.sday_roll_max) if pd.notna(r.sday_roll_max) else np.nan,
                "json_file_path": str(r.json_file_path),
                "series_rows": r.series_rows if isinstance(r.series_rows, list) else [],
            }
        )
    for t in tqdm(export_tasks, total=len(export_tasks), desc="Export JSON"):
        _export_result_json(t, output_root=output_root, params_payload=params_payload)

    # Build direct-read cache map for CropSeasonFirstSporeInfThr:
    # key -> "{variety}_{year}_{grid_id}", value -> {date: S_day}
    spore_day_map: Dict[str, Dict[str, float]] = {}
    spore_day_roll_map: Dict[str, Dict[str, float]] = {}
    cache_rows: List[Dict] = []
    for r in export_tasks:
        key = f"{str(r['variety']).upper()}_{int(r['year'])}_{int(r['grid_id'])}"
        series_rows = r.get("series_rows", []) or []
        day_map: Dict[str, float] = {}
        day_roll_map: Dict[str, float] = {}
        for row in series_rows:
            d = str(row.get("date", "")).strip()
            if not d:
                continue
            try:
                s_day = float(row.get("S_day", 0.0))
            except Exception:
                s_day = 0.0
            try:
                s_roll = float(row.get("S_day_roll", 0.0))
            except Exception:
                s_roll = 0.0
            day_map[d] = s_day
            day_roll_map[d] = s_roll
            cache_rows.append(
                {
                    "key": key,
                    "variety": str(r["variety"]).upper(),
                    "year": int(r["year"]),
                    "grid_id": int(r["grid_id"]),
                    "date": d,
                    "S_day": s_day,
                    "S_day_roll": s_roll,
                }
            )
        spore_day_map[key] = day_map
        spore_day_roll_map[key] = day_roll_map

    cache_payload = {
        "index_key_spec": "VARIETY_YEAR_GRIDID (e.g., MPT_2021_123)",
        "value_spec": "date->S_day",
        "spore_day_map": spore_day_map,
        "spore_day_roll_map": spore_day_roll_map,
    }
    pkl_path = output_root / "spore_day_map.pkl"
    pd.to_pickle(cache_payload, pkl_path)
    print(f"Saved: {pkl_path}")

    # Optional parquet for easy querying/debugging.
    parquet_path = output_root / "spore_day_map.parquet"
    try:
        cache_df = pd.DataFrame(cache_rows)
        cache_df.to_parquet(parquet_path, index=False)
        print(f"Saved: {parquet_path}")
    except Exception as e:
        print(f"[WARN] Skip parquet export ({parquet_path}): {e}")

    run_cfg = {
        "source": "invert_spore_pressure_historical.py",
        "input_root": str(input_root),
        "output_root": str(output_root),
        "params_json": str(params_path),
        "params_used": params_payload,
        "year_min": int(args.year_min),
        "year_max": int(args.year_max),
        "varieties": [VARIETY_CODE_MAP[v] for v in var_list if v in VARIETY_CODE_MAP],
        "backend": str(args.backend),
        "n_jobs": int(args.n_jobs),
        "cache_files": {
            "spore_day_map_pkl": "spore_day_map.pkl",
            "spore_day_map_parquet": "spore_day_map.parquet",
        },
    }
    (output_root / "spore_pressure_inversion_config.json").write_text(json.dumps(run_cfg, indent=2), encoding="utf-8")

    print(f"Saved: {output_root / 'spore_pressure_all_varieties_2000_2024.csv'}")
    print(f"Saved: {output_root / 'spore_pressure_inversion_config.json'}")


if __name__ == "__main__":
    main()
