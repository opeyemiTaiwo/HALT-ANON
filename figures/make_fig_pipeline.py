import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
plt.rcParams.update({'font.family':'serif','savefig.bbox':'tight'})

fig, ax = plt.subplots(figsize=(6.9, 3.15))
UX = 10.0
INH, NEW, ACC, ED = '#eceff3', '#ccdaea', '#3f6193', '#59677a'
FS = 6.2
BW, GAP = 1.80, 0.25
xs = [i*(BW+GAP) for i in range(5)]

YA, YB, YC = 2.62, 1.44, 0.22          # three tiers
HA, HB, HC = 0.80, 0.58, 0.86

def box(x, y, w, h, lines, fc, tc='black', fs=FS):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.015",
                                facecolor=fc, edgecolor=ED, linewidth=0.7))
    ax.text(x+w/2, y+h/2, "\n".join(lines), ha='center', va='center',
            fontsize=fs, color=tc, linespacing=1.45)

def arr(x1, y1, x2, y2, rad=0.0):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>',
                 mutation_scale=8, color=ED, linewidth=0.8,
                 shrinkA=0, shrinkB=0, connectionstyle=f"arc3,rad={rad}"))

def tag(x, y, s, ha='center'):
    ax.text(x, y, s, fontsize=5.6, style='italic', color='0.42',
            ha=ha, va='center')

# band shading
ax.add_patch(Rectangle((-0.12, YA-0.20), UX+0.24, HA+0.40,
                       facecolor='#f6f7f9', edgecolor='none', zorder=0))
ax.text(-0.12, YA+HA+0.32, 'Inherited: corpus construction',
        fontsize=6.4, style='italic', color='0.40')
ax.text(-0.12, YC+HC+0.34, 'This paper: labels, splits, detection',
        fontsize=6.4, style='italic', color='0.40')

# ---- tier A, left to right
top = [['807 UCF-Crime', 'videos + UCA', 'reference annot.'],
       ['3 source LLMs', '\u00d7 8 prompting', 'strategies'],
       ['19,361 forensic', 'reports', '(21.8M words)'],
       ['cross-judging', 'panel: 2 of 3,', 'no self-evaluation'],
       ['38,722 per-judge', 'H1\u2013H6 verdicts']]
for x, l in zip(xs, top):
    box(x, YA, BW, HA, l, INH)
for i in range(4):
    arr(xs[i]+BW, YA+HA/2, xs[i+1], YA+HA/2)

# ---- tier B: human validation, fed by the report corpus
box(xs[3], YB, BW, HB, ['130-report subset,', '2 human raters'], NEW)
arr(xs[2]+BW*0.72, YA, xs[3]+0.30, YB+HB, rad=-0.12)
tag(xs[2]+BW+0.42, YB+HB+0.30, 'stratified subset')

# ---- tier C, right to left
bot = [['per-type AUC,', 'macro, and', 'bootstrap CIs'],
       ['5 feature sets:', 'surface, lexical,', 'embedding,', 'reference-cond.'],
       ['4 video-disjoint', 'splits, incl.', 'leave-one-', 'source-out'],
       ['two-judge', 'aggregation +', 'reliability tiers']]
bx = [xs[1], xs[2], xs[3], xs[4]]
for x, l in zip(bx, bot):
    box(x, YC, BW, HC, l, NEW)
for i in range(3, 0, -1):
    arr(bx[i], YC+HC/2, bx[i-1]+BW, YC+HC/2)

# verdicts down into aggregation (enters right of centre)
arr(xs[4]+BW*0.70, YA, xs[4]+BW*0.70, YC+HC)
# human labels calibrate the tiers (enters left of centre, no corner clipping)
arr(xs[3]+BW, YB+HB*0.5, xs[4]+BW*0.24, YC+HC, rad=-0.30)
tag(xs[3]+BW+0.16, YB+HB*0.82, 'calibrates', ha='left')

# ---- outcome
box(xs[0], YC+0.14, BW*0.72, HC-0.28, ['RQ1\u2013RQ5'], ACC, tc='white', fs=7.4)
arr(bx[0], YC+HC/2, xs[0]+BW*0.72, YC+HC/2)

ax.set_xlim(-0.15, UX+0.15); ax.set_ylim(-0.10, YA+HA+0.50); ax.axis('off')
fig.savefig('fig_pipeline.pdf'); fig.savefig('fig_pipeline.png', dpi=190)
print('rendered')
