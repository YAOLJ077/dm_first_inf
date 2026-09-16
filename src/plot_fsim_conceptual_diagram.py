#!/usr/bin/env python3
"""Draw current FSIM at 180 mm double-column width."""
from pathlib import Path
import os, hashlib, json
os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/fsim_conceptual_mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch, FancyArrowPatch
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'fig/fsim_conceptual_diagram_current'


def main():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'pdf.fonttype':42,'svg.fonttype':'none'})
    fig=plt.figure(figsize=(180/25.4,164/25.4))
    ax=fig.add_axes([0,0,1,1]); ax.set(xlim=(0,180),ylim=(0,164)); ax.axis('off')
    colors={'state':('#EAF2F8','#527F9F'),'process':('#F7EEDB','#B69552'),
            'driver':('#EDF4F0','#729784'),'loss':('#FAEEEB','#BB8178')}
    checks=[]
    def box(x,y,w,h,label,kind):
        fill,edge=colors[kind]
        if kind=='state':
            patch=Rectangle((x,y),w,h,facecolor=fill,edgecolor=edge,lw=.7)
        else:
            patch=FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0,rounding_size=1.5',facecolor=fill,edgecolor=edge,lw=.7)
        ax.add_patch(patch)
        txt=ax.text(x+w/2,y+h/2,label,ha='center',va='center',fontsize=8,
                    fontweight='bold' if kind in ('state','process') else 'normal',color='#243747',linespacing=1.3)
        checks.append((txt,patch))
    def arrow(a,b,dashed=False,color='#243747'):
        ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=8,linewidth=.7,color=color,
                                    linestyle=(0,(3,2)) if dashed else '-',shrinkA=1.5,shrinkB=1.5))
    processes=['1. Sporulation','2. Dispersal and deposition','3. Zoospore release','4. Infection','5. Incubation']
    states=['Attached sporangia\nRelative dose','Detached sporangia\nRelative leaf availability',
            'Viable zoospores\nSurvival window','Infection time','6. Oil spot']
    drivers=['Night-time T, RH, rain\nOptional leaf wetness\nMP, TMP','Wind speed\nOptional gate',
             'VPD, WD, TWD','VPD, WD, TWD\nBBCH15','Temperature']
    for i in range(5):
        y=148-32*i
        box(62,y,60,12,processes[i],'process'); box(62,y-16,60,12,states[i],'state')
        arrow((92,y),(92,y-4))
        if i<4: arrow((92,y-16),(92,y-20))
        box(3,y-2,49,16,drivers[i],'driver')
        arrow((52,y+6),(62,y+6),True,colors['driver'][1])
    # Outgoing branches distinguish termination from transient waiting.
    # Missing an incubation endpoint at the data boundary is incomplete output,
    # not biological death. Dryness after infection does not cancel incubation.
    exits=[
        (154,'Production = 0','No cohort','loss'),
        (138,'Cumulative\nmortality ≥ 1','Terminated','loss'),
        (106,'Mortality ≥ 1\nbefore release','Terminated','loss'),
        (90,'No release\nbefore expiry','No event','loss'),
        (74,'Dry before\ninfection','Terminated','loss'),
        (58,'Requirement unmet\nbefore expiry','No event','loss'),
    ]
    for source_y,condition,outcome,kind in exits:
        box(159,source_y-4.5,19,9,outcome,kind)
        arrow((122,source_y),(159,source_y),False,colors[kind][1])
        # Conditions annotate branches; boxes contain only their outcomes.
        ax.text(140.5,source_y+1.2,condition,ha='center',va='bottom',
                fontsize=7,color='#243747',linespacing=1.15)
    fig.canvas.draw(); renderer=fig.canvas.get_renderer()
    for txt,patch in checks:
        a,b=txt.get_window_extent(renderer),patch.get_window_extent(renderer)
        if a.x0<b.x0+1 or a.x1>b.x1-1 or a.y0<b.y0+1 or a.y1>b.y1-1:
            raise ValueError('Text exceeds box: '+txt.get_text())
    for ext in ('png','pdf','svg'): fig.savefig(OUT.with_suffix('.'+ext),dpi=600,facecolor='white')
    plt.close(fig)
    sources=['model/first_infection.py','model/crop_season.py','model/sporulation_timing.py','model/process_rules.py']
    OUT.with_suffix('.json').write_text(json.dumps({'scope':'Current baseline FSIM; concise conceptual diagram',
        'width_mm':180,'height_mm':164,'font_pt':8,'png_dpi':600,
        'source_sha256':{s:hashlib.sha256((ROOT/s).read_bytes()).hexdigest() for s in sources}},indent=2)+'\n')
    print(f'Saved {OUT}: PNG, PDF, SVG; 180 x 164 mm.')
if __name__=='__main__': main()
