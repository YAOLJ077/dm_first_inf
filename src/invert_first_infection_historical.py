#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
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

from model.config import config as base_config
from model.crop_season_spore_infthr import CropSeasonFirstSporeInfThr

os.environ.setdefault("RAY_DISABLE_METRICS", "1")
os.environ.setdefault("RAY_DISABLE_USAGE_STATS", "1")
os.environ.setdefault("RAY_METRICS_EXPORT_PORT", "0")
os.environ.setdefault("RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO", "0")


def _apply_params(base_cfg: dict, values: Dict[str, float]) -> dict:
    cfg = json.loads(json.dumps(base_cfg))
    for k, v in values.items():
        if "__" not in str(k):
            continue
        group, sub = str(k).split("__", 1)
        if group not in cfg:
            cfg[group] = {}
        cfg[group][sub] = float(v)
    return cfg


def _load_model_config(params_json_path: Path) -> dict:
    if not params_json_path.exists():
        raise FileNotFoundError(f"Missing model params json: {params_json_path}")
    payload = json.loads(params_json_path.read_text(encoding="utf-8"))
    params = payload.get("params", None)
    if params is None:
        raise KeyError(f"Missing key 'params' in {params_json_path}")
    if not isinstance(params, dict) or not params:
        raise ValueError(f"Invalid/empty params in {params_json_path}")
    return _apply_params(base_config, params)


def _load_spore_day_from_grid_json(spore_root: Path, variety: str, year: int, grid_id: int) -> Optional[pd.Series]:
    p = spore_root / str(variety).lower() / str(int(year)) / f"grid_{int(grid_id)}_spore_pressure.json"
    if not p.exists():
        return None
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
        rows = payload.get("spore_day_series", [])
        if not rows:
            return None
        df = pd.DataFrame(rows)
        if "date" not in df.columns or "S_day" not in df.columns:
            return None
        idx = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
        vals = pd.to_numeric(df["S_day"], errors="coerce")
        s = pd.Series(vals.values, index=idx).dropna()
        return s if not s.empty else None
    except Exception:
        return None


def _build_tasks(
    bbch_csv: Path,
    input_root: Path,
    year_min: int,
    year_max: int,
    varieties: List[str],
) -> List[Dict]:
    if not bbch_csv.exists():
        raise FileNotFoundError(f"Missing BBCH CSV: {bbch_csv}")
    df = pd.read_csv(bbch_csv)
    required = {"variety", "year", "grid_id", "latitude", "longitude", "bbch53_date"}
    miss = [c for c in required if c not in df.columns]
    if miss:
        raise KeyError(f"BBCH CSV missing columns: {miss}")

    df["variety"] = df["variety"].astype(str).str.upper()
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df["grid_id"] = pd.to_numeric(df["grid_id"], errors="coerce").astype("Int64")
    df["bbch53_date"] = pd.to_datetime(df["bbch53_date"], errors="coerce")
    df = df.dropna(subset=["year", "grid_id", "bbch53_date"]).copy()
    df["year"] = df["year"].astype(int)
    df["grid_id"] = df["grid_id"].astype(int)
    df = df[(df["year"] >= int(year_min)) & (df["year"] <= int(year_max))].copy()
    if varieties:
        keep = {v.upper() for v in varieties}
        df = df[df["variety"].isin(keep)].copy()

    tasks: List[Dict] = []
    for r in df.itertuples(index=False):
        weather_path = None
        if "json_abs_path" in df.columns:
            p = Path(str(getattr(r, "json_abs_path")))
            if p.exists():
                weather_path = p
        if weather_path is None:
            if "json_file_path" not in df.columns:
                continue
            rel = str(getattr(r, "json_file_path"))
            weather_path = input_root / str(r.variety).lower() / rel
            if not weather_path.exists():
                continue
        tasks.append(
            {
                "variety": str(r.variety).upper(),
                "year": int(r.year),
                "grid_id": int(r.grid_id),
                "latitude": float(r.latitude),
                "longitude": float(r.longitude),
                "bbch53_date": pd.to_datetime(r.bbch53_date).normalize().strftime("%Y-%m-%d"),
                "weather_json": str(weather_path),
            }
        )
    return tasks


def _run_one(task: Dict, cfg: Dict, spore_root: str) -> Dict:
    variety = str(task["variety"]).upper()
    year = int(task["year"])
    grid_id = int(task["grid_id"])
    weather_json = Path(str(task["weather_json"]))
    bbch53_date = str(task["bbch53_date"])

    if not weather_json.exists():
        return {
            **task,
            "infection_datetime": "",
            "oil_spot_start": "",
            "oil_spot_end": "",
            "has_event": False,
            "status": "missing_weather_json",
            "has_spore": False,
        }
    try:
        raw = json.loads(weather_json.read_text(encoding="utf-8"))
        raw["variety_code"] = variety
        raw["Place"] = raw.get("Place", f"grid_{grid_id}")
        raw["BBCH_53"] = bbch53_date
        raw["HHCH_53"] = bbch53_date

        spore_day = _load_spore_day_from_grid_json(Path(spore_root), variety, year, grid_id)
        has_spore = spore_day is not None and not spore_day.empty

        cs = CropSeasonFirstSporeInfThr(csrb=raw, config=cfg, spore_day=spore_day)
        cs.simulate()
        ev = cs.primary_chain_unique
        if ev is None or ev.empty:
            return {
                **task,
                "infection_datetime": "",
                "oil_spot_start": "",
                "oil_spot_end": "",
                "has_event": False,
                "status": "no_event",
                "has_spore": bool(has_spore),
            }
        first = ev.iloc[0]
        infection_dt = pd.to_datetime(first.get("infection_datetime"), errors="coerce")
        oil_start = pd.to_datetime(first.get("oil_spot_start"), errors="coerce")
        oil_end = pd.to_datetime(first.get("oil_spot_end"), errors="coerce")
        return {
            **task,
            "infection_datetime": infection_dt.strftime("%Y-%m-%d %H:%M:%S") if pd.notna(infection_dt) else "",
            "oil_spot_start": oil_start.strftime("%Y-%m-%d %H:%M:%S") if pd.notna(oil_start) else "",
            "oil_spot_end": oil_end.strftime("%Y-%m-%d %H:%M:%S") if pd.notna(oil_end) else "",
            "has_event": True,
            "status": "ok",
            "has_spore": bool(has_spore),
        }
    except Exception as e:  # pragma: no cover
        return {
            **task,
            "infection_datetime": "",
            "oil_spot_start": "",
            "oil_spot_end": "",
            "has_event": False,
            "status": f"error:{type(e).__name__}",
            "has_spore": False,
        }


if ray is not None:
    @ray.remote
    def _run_one_ray(task: Dict, cfg: Dict, spore_root: str) -> Dict:
        return _run_one(task, cfg, spore_root)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Historical first-infection inversion by variety-year-grid using BBCH53 and spore pressure inputs.")
    p.add_argument("--bbch-csv", type=str, default="../bbch53_date/bbch53_dates_all_varieties_2000_2024.csv")
    p.add_argument("--input-root", type=str, default="../input_weather_json")
    p.add_argument("--spore-root", type=str, default="../spore_pressure")
    p.add_argument("--params-json", type=str, default="result/calibration_sum/final_spore_params.json")
    p.add_argument("--output-root", type=str, default="../infection_date_2000_2024")
    p.add_argument("--varieties", type=str, default="wk,jf,mpt")
    p.add_argument("--year-min", type=int, default=2000)
    p.add_argument("--year-max", type=int, default=2024)
    p.add_argument("--backend", type=str, default="ray", choices=["ray", "process", "serial"])
    p.add_argument("--n-jobs", type=int, default=256)
    p.add_argument("--chunksize", type=int, default=16)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    bbch_csv = Path(args.bbch_csv).expanduser()
    if not bbch_csv.is_absolute():
        bbch_csv = (PROJECT_ROOT / bbch_csv).resolve()
    input_root = Path(args.input_root).expanduser()
    if not input_root.is_absolute():
        input_root = (PROJECT_ROOT / input_root).resolve()
    spore_root = Path(args.spore_root).expanduser()
    if not spore_root.is_absolute():
        spore_root = (PROJECT_ROOT / spore_root).resolve()
    params_json = Path(args.params_json).expanduser()
    if not params_json.is_absolute():
        params_json = (PROJECT_ROOT / params_json).resolve()
    output_root = Path(args.output_root).expanduser()
    if not output_root.is_absolute():
        output_root = (PROJECT_ROOT / output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    cfg = _load_model_config(params_json)
    varieties = [v.strip().upper() for v in str(args.varieties).split(",") if v.strip()]
    tasks = _build_tasks(
        bbch_csv=bbch_csv,
        input_root=input_root,
        year_min=int(args.year_min),
        year_max=int(args.year_max),
        varieties=varieties,
    )
    if not tasks:
        print("[WARN] No tasks generated.")
        return

    rows: List[Dict] = []
    backend = str(args.backend).lower()
    if backend == "serial":
        for t in tqdm(tasks, total=len(tasks), desc="Infection inversion"):
            rows.append(_run_one(t, cfg=cfg, spore_root=str(spore_root)))
    elif backend == "process":
        max_workers = max(1, min(int(args.n_jobs), int(os.cpu_count() or 1), 61 if os.name == "nt" else 9999))
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            futs = [ex.submit(_run_one, t, cfg, str(spore_root)) for t in tasks]
            for fut in tqdm(as_completed(futs), total=len(futs), desc="Infection inversion"):
                rows.append(fut.result())
    else:
        if ray is None:
            print("[WARN] backend=ray but ray not installed, fallback to process.")
            max_workers = max(1, min(int(args.n_jobs), int(os.cpu_count() or 1), 61 if os.name == "nt" else 9999))
            with ProcessPoolExecutor(max_workers=max_workers) as ex:
                futs = [ex.submit(_run_one, t, cfg, str(spore_root)) for t in tasks]
                for fut in tqdm(as_completed(futs), total=len(futs), desc="Infection inversion"):
                    rows.append(fut.result())
        else:
            ncpu = max(1, min(int(args.n_jobs), int(os.cpu_count() or 1)))
            ray.init(num_cpus=ncpu, include_dashboard=False, logging_level="ERROR", ignore_reinit_error=True)
            futs = [_run_one_ray.remote(t, cfg, str(spore_root)) for t in tasks]
            pending = list(futs)
            pbar = tqdm(total=len(futs), desc="Infection inversion")
            while pending:
                ready, pending = ray.wait(pending, num_returns=1)
                rows.append(ray.get(ready[0]))
                pbar.update(1)
            pbar.close()
            ray.shutdown()

    out = pd.DataFrame(rows).sort_values(["variety", "year", "grid_id"]).reset_index(drop=True)
    out_all = output_root / "infection_inversion_all_varieties_2000_2024.csv"
    out.to_csv(out_all, index=False)

    for v in sorted(out["variety"].dropna().astype(str).unique()):
        out_v = out[out["variety"] == v].copy()
        out_dir_v = output_root / str(v).lower()
        out_dir_v.mkdir(parents=True, exist_ok=True)
        out_v.to_csv(out_dir_v / "infection_inversion_all_years.csv", index=False)
        for y, sub in out_v.groupby("year"):
            sub.to_csv(out_dir_v / f"infection_inversion_{int(y)}.csv", index=False)

    no_inf = out[~out["has_event"].astype(bool)].copy()
    no_inf = no_inf[["variety", "year", "grid_id", "latitude", "longitude", "bbch53_date", "status", "has_spore", "weather_json"]]
    no_inf_path = output_root / "infection_no_event_variety_year_grid.csv"
    no_inf.to_csv(no_inf_path, index=False)

    summary = (
        out.groupby(["variety", "year"], as_index=False)
        .agg(
            n_total=("grid_id", "count"),
            n_event=("has_event", "sum"),
            n_no_event=("has_event", lambda x: int((~x.astype(bool)).sum())),
            n_error=("status", lambda x: int(pd.Series(x).astype(str).str.startswith("error:").sum())),
            n_missing_spore=("has_spore", lambda x: int((~x.astype(bool)).sum())),
        )
        .sort_values(["variety", "year"])
        .reset_index(drop=True)
    )
    summary_path = output_root / "infection_inversion_summary_by_variety_year.csv"
    summary.to_csv(summary_path, index=False)

    run_cfg = {
        "source": "invert_first_infection_historical.py",
        "params_json": str(params_json),
        "bbch_csv": str(bbch_csv),
        "input_root": str(input_root),
        "spore_root": str(spore_root),
        "output_root": str(output_root),
        "varieties": varieties,
        "year_min": int(args.year_min),
        "year_max": int(args.year_max),
        "backend": str(args.backend),
        "n_jobs": int(args.n_jobs),
        "n_tasks": int(len(tasks)),
        "files": {
            "all": out_all.name,
            "no_event": no_inf_path.name,
            "summary": summary_path.name,
        },
    }
    (output_root / "infection_inversion_config.json").write_text(json.dumps(run_cfg, indent=2), encoding="utf-8")

    print(f"Saved: {out_all}")
    print(f"Saved: {no_inf_path}")
    print(f"Saved: {summary_path}")
    print(f"Saved: {output_root / 'infection_inversion_config.json'}")


if __name__ == "__main__":
    main()
