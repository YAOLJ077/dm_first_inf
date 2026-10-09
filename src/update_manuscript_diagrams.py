#!/usr/bin/env python3
"""Align manuscript diagram labels with the implemented FSIM processes."""

from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
BACKUP = Path('/private/tmp/pms_review/original_figures')
FONT = '/System/Library/Fonts/Supplemental/Times New Roman.ttf'


def source_image(name):
    BACKUP.mkdir(parents=True, exist_ok=True)
    backup = BACKUP / name
    if not backup.exists():
        shutil.copy2(ROOT / 'fig' / name, backup)
    return Image.open(backup).convert('RGB')


def label(draw, box, text, size, fill, background):
    draw.rectangle(box, fill=background)
    font = ImageFont.truetype(FONT, size)
    while draw.textbbox((0, 0), text, font=font)[2] > box[2] - box[0] - 6:
        size -= 1
        font = ImageFont.truetype(FONT, size)
    draw.text((box[0] + 3, box[1]), text, font=font, fill=fill)


def update_workflow():
    image = source_image('figure_01.png')
    draw = ImageDraw.Draw(image)
    label(draw, (34, 522, 319, 555), 'dew point; DPD wetness proxy', 28, '#666666', 'white')
    label(draw, (408, 665, 718, 698), 'Cohort gating + dose input', 28, '#666666', 'white')
    image.save(ROOT / 'fig/figure_01.png')


def update_management():
    image = source_image('figure_07.png')
    draw = ImageDraw.Draw(image)
    # Retain the marker at DOY 128, the left edge of the observed interval.
    left, right, top, bottom = 1050, 1990, 143, 202
    draw.rectangle((left, top, 2150, bottom), fill='white')
    draw.rectangle((left, top, 1078, bottom), fill='#efefef')
    label(draw, (left + 6, top, right, bottom), 'Symptom-window start (DOY 128)', 46, '#8b0000', 'white')
    image.save(ROOT / 'fig/figure_07.png')


def draw_processes():
    source_image('figure_06.png')
    plt.rcParams.update({'font.family': 'Times New Roman', 'font.size': 12,
                         'mathtext.fontset': 'stix'})
    fig, ax = plt.subplots(figsize=(10.5, 12.6))
    ax.set(xlim=(0, 11.7), ylim=(-0.05, 14.1))
    ax.axis('off')

    def box(x, y, width, height, text, *, shade='white', size=15, bold=False):
        ax.add_patch(Rectangle((x - width / 2, y - height / 2), width, height,
                               facecolor=shade, edgecolor='black', linewidth=1.1))
        ax.text(x, y, text, ha='center', va='center', fontsize=size,
                fontweight='bold' if bold else 'normal', linespacing=1.3)

    def arrow(start, end, *, dashed=False):
        ax.annotate('', xy=end, xytext=start,
                    arrowprops={'arrowstyle': '-|>', 'lw': 1.0, 'color': 'black',
                                'linestyle': '--' if dashed else '-'})

    x, width = 5.7, 3.5
    stages = [
        (12.7, 'Recurrent sporangial cohorts\nMoist-night sporulation'),
        (11.0, 'Attached sporangia\nAttached-state survival'),
        (9.3, 'Airborne sporangia\nDetached-state survival'),
        (7.6, 'Deposited sporangia\nDetached-state survival'),
        (5.9, 'Zoospore release\nand wetness-dependent survival'),
        (4.2, 'Infection\nWetness and thermal requirement'),
        (2.5, 'Incubation\nTemperature-dependent progress'),
        (0.8, 'First visible symptoms\nOnset window'),
    ]
    ax.text(x, 13.75, 'Shared FSIM / FSIM-S process chain', ha='center',
            fontsize=16, fontweight='bold')
    for index, (y, text) in enumerate(stages):
        box(x, y, width, 1.0, text, shade='#f4f4f4')
        if index:
            arrow((x, stages[index - 1][0] - 0.5), (x, y + 0.5))

    drivers = [
        (12.7, 'Night-time moisture\nTemperature, RH, rainfall'),
        (9.3, 'Wind dispersal\nWind-speed threshold'),
        (7.6, 'Rainfall-dependent deposition\nHeavy-rain wash-off'),
        (5.9, 'Temperature and wet duration\nDPD / rainfall wetness proxy'),
        (4.2, 'Host susceptibility\nPhenological stage'),
        (2.5, 'Air temperature'),
    ]
    for y, text in drivers:
        box(1.7, y, 3.0, 0.95, text, size=14)
        arrow((3.2, y), (x - width / 2, y))

    box(9.8, 12.7, 3.0, 1.25,
        'FSIM-S observation constraint\nZero-count cohort gating\nPositive-pressure dose input',
        shade='#e7eef3', size=14)
    arrow((8.3, 12.7), (x + width / 2, 12.7), dashed=True)
    box(9.8, 10.0, 3.0, 1.2,
        'Sporangial mortality\nEmpirical index\n' + r'$x=T(1-\mathrm{RH}/100)$', size=14)
    for target_y in (11.0, 9.3, 7.6):
        ax.plot([8.1, 8.1], [target_y, 10.0], color='black', lw=0.9,
                linestyle='--')
        arrow((8.1, target_y), (x + width / 2, target_y), dashed=True)
    ax.plot([8.1, 8.3], [10.0, 10.0], color='black', lw=0.9, linestyle='--')
    box(9.8, 5.9, 3.0, 1.2,
        'Zoospore survival\nElapsed time / wet hours', size=14)
    arrow((8.3, 5.9), (x + width / 2, 5.9), dashed=True)
    ax.text(9.8, 3.6, 'Solid arrows: process / driver\nDashed arrows: dose or survival\nconstraints',
            ha='center', va='center', fontsize=13, linespacing=1.5)
    fig.subplots_adjust(left=0.025, right=0.995, bottom=0.02, top=0.99)
    fig.savefig(ROOT / 'fig/figure_06.png', dpi=300, facecolor='white')
    plt.close(fig)


if __name__ == '__main__':
    update_workflow()
    update_management()
    draw_processes()
