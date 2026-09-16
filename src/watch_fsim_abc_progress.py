#!/usr/bin/env python3
"""Read-only progress panel for an already running A/B/C calibration."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def bar(done, total):
    ratio = min(1, max(0, done / total)) if total else 0
    filled = int(ratio * 24)
    return f'[{"#" * filled}{"-" * (24 - filled)}] {ratio:6.1%}  {done}/{total}'


def render(settings, metadata, live):
    budget = settings.get('n_trials', metadata.get('target_trials', 1024))
    order = settings.get('order', ['A', 'B', 'C'])
    phase = live.get('phase', '等待进度文件')
    status = metadata.get('status', '未知')
    counts = {job: value.get('n_trials', 0) for job, value in metadata.get('jobs', {}).items()}
    for study in live.get('studies', []):
        job = study['job']
        counts[job] = max(counts.get(job, 0), study.get('completed_trials', 0))
    totals = {plan: sum(min(budget, n) for job, n in counts.items()
                        if job.split('/')[0] == plan) for plan in order}
    lines = ['FSIM A → B → C 进度', f'运行状态：{status}    当前阶段：{phase}',
             f'最后更新：{live.get("utc", "暂无")}', '',
             '总训练进度  ' + bar(sum(totals.values()), len(order) * 12 * budget)]
    for plan in order:
        lines.append(f'{plan} 训练进度  ' + bar(totals[plan], 12 * budget))
    plan = phase.split('_')[0]
    studies = live.get('studies', [])
    if studies:
        lines.extend(['', f'{plan} 各折已完成 trial：'])
        for scheme, label in [('losyo', 'LOSYO'), ('site', '留出站点'), ('year', '留出年份')]:
            parts = []
            for study in studies:
                pieces = study['job'].split('/')
                if len(pieces) > 1 and pieces[1] == scheme:
                    n = counts[study['job']]
                    parts.append(f'{pieces[-1]} {n}/{budget} ({n / budget:.1%})')
            lines.append(f'  {label}：' + ' | '.join(parts))
        lines.extend(['', '训练百分比按已反馈的 trial 统计；启动批次完成后会跳跃更新。'])
    elif live:
        lines.extend(['', '当前为评估/选优阶段，旧版进度文件未记录可用的完成百分比分母。'])
    lines.append('本次进程累计模型计算：{}  缓存命中：{}  已提交未回收：{}  待提交：{}'.format(
        live.get('completed', 0), live.get('cache_hits', 0),
        live.get('in_flight_tasks', 0), live.get('ready_requests', 0)))
    lines.extend(['', '训练 100% 不代表评估和探索性选优已完成；运行状态为 complete 才表示全部完成。',
                  'Ctrl+C 仅退出此监控面板。'])
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'result/fsim_abc_tpe_seed123')
    parser.add_argument('--interval', type=float, default=5)
    parser.add_argument('--once', action='store_true', help='Print one snapshot and exit')
    args = parser.parse_args()
    if args.interval <= 0:
        parser.error('--interval must be positive')
    try:
        while True:
            settings = read_json(args.output_dir / 'effective_settings.json')
            metadata = read_json(args.output_dir / 'run_metadata.json')
            live = read_json(args.output_dir / 'live_progress.json')
            if sys.stdout.isatty() and not args.once:
                print('\033[2J\033[H', end='')
            print(render(settings, metadata, live), flush=True)
            if args.once or metadata.get('status') == 'complete':
                break
            time.sleep(args.interval)
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
