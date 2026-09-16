#!/usr/bin/env python3
"""Read-only terminal monitor for scheme-B robust TPE, including resumed runs.

Uses only the Python standard library. Copy this file to the server's src/
directory; no changes to the running calibration are required.
"""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
SCHEMES = [('losyo', 'LOSYO', 6), ('site', '留出站点', 3), ('year', '留出年份', 3)]


def read_json(path):
    try:
        value = json.loads(path.read_text())
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def bar(done, total):
    ratio = min(1, max(0, done / total)) if total else 0
    filled = int(ratio * 24)
    return f'[{"#" * filled}{"-" * (24 - filled)}] {ratio:6.1%}  {done}/{total}'


def timestamp(value):
    try:
        dt = datetime.fromisoformat(value.replace('Z', '+00:00'))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except (ValueError, TypeError, AttributeError):
        return None


def render(settings, metadata, live, stale_seconds=120):
    budget = metadata.get('target_trials', settings.get('n_trials', 2048))
    status = metadata.get('status', '未知')
    phase = live.get('phase', '等待进度文件')
    counts = {job: value.get('n_trials', 0)
              for job, value in metadata.get('jobs', {}).items() if job.startswith('B/')}
    for study in live.get('studies', []):
        job = study.get('job', '')
        if job.startswith('B/'):
            counts[job] = max(counts.get(job, 0), study.get('completed_trials', 0))
    lines = ['FSIM 方案 B 升级版（2p）进度',
             f'记录状态：{status}    当前阶段：{phase}',
             f'每折预算：{budget}（含启动候选 {settings.get("n_startup_trials", 248)}）',
             f'最后心跳（UTC）：{live.get("utc", "暂无")}']
    now = datetime.now(timezone.utc)
    updated = timestamp(live.get('utc'))
    if updated:
        age = max(0, (now - updated).total_seconds())
        lines.append(f'心跳距今：{age:.0f} 秒')
        if age > stale_seconds and status != 'complete':
            lines.append('注意：进度文件已过期；可能中断、同步延迟或正在恢复，不能据此认定进程存活。')
    sessions = metadata.get('sessions', [])
    if sessions:
        session = sessions[-1]
        lines.append(f'已记录启动次数：{len(sessions)}    最近启动（UTC）：{session.get("started_utc", "未知")}')
        started = timestamp(session.get('started_utc'))
        if started and updated and updated < started:
            lines.append('当前心跳早于最近启动：仍显示上次进度，等待续跑恢复后的新心跳。')
    if not settings and not metadata and not live:
        lines.append('尚未读到有效记录，请检查输出目录或等待校准写入进度。')
    lines.extend(['', '总训练进度  ' + bar(sum(min(budget, n) for n in counts.values()), 12 * budget)])
    for scheme, label, folds in SCHEMES:
        rows = sorted((job.split('/')[-1], n) for job, n in counts.items()
                      if job.split('/')[1] == scheme)
        lines.append(f'{label}  ' + bar(sum(min(budget, n) for _, n in rows), folds * budget))
        for name, n in rows:
            lines.append(f'  {name:<18} {bar(n, budget)}')
        if len(rows) < folds:
            lines.append(f'  还有 {folds - len(rows)} 折尚无进度记录。')
    lines.extend(['',
        '本次调度累计模型计算：{}    缓存命中：{}'.format(live.get('completed', 0), live.get('cache_hits', 0)),
        '已提交未回收：{}    待提交请求：{}    曾出现的工作进程数：{}'.format(
            live.get('in_flight_tasks', 0), live.get('ready_requests', 0), live.get('worker_pids_seen', 0)),
        '累计吞吐：{:.2f} 次模型计算/秒'.format(live.get('completed_model_tasks_per_second', 0)),
        '',
        '训练计数包含已恢复的 trial；模型计算及缓存计数属于本次调度。',
        '启动批次或恢复期间计数可能暂时不变；已提交任务数不等于正在占用 CPU 的进程数。',
        '训练 100% 后仍可能进行留出评估和探索性选优，记录状态 complete 才表示全部完成。',
        '此面板只读取进度文件；Ctrl+C 仅退出监控，不停止校准。'])
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'result/fsim_b_robust_tpe_248_2048')
    parser.add_argument('--interval', type=float, default=5, help='Refresh interval in seconds')
    parser.add_argument('--stale-seconds', type=float, default=120, help='Heartbeat age warning threshold')
    parser.add_argument('--once', action='store_true', help='Print one snapshot and exit')
    args = parser.parse_args()
    for name in ('interval', 'stale_seconds'):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(f'--{name.replace("_", "-")} must be finite and positive')
    try:
        while True:
            settings = read_json(args.output_dir / 'effective_settings.json')
            metadata = read_json(args.output_dir / 'run_metadata.json')
            live = read_json(args.output_dir / 'live_progress.json')
            if sys.stdout.isatty() and not args.once:
                print('\033[2J\033[H', end='')
            print(f'结果目录：{args.output_dir.resolve()}')
            print(render(settings, metadata, live, args.stale_seconds), flush=True)
            if args.once or metadata.get('status') == 'complete':
                break
            time.sleep(args.interval)
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
