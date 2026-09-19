#!/usr/bin/env python3
"""Figure 1: the HALT pipeline, from raw video to the five research questions.

Upper band is inherited from the prior study; lower band is this paper. The
layout follows the data: videos become reports, reports become verdicts,
verdicts become labels, labels are partitioned, detectors are scored on them.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

INK, ACC = '#16334f', '#2d5d8a'
TAN, TAN_E = '#f4e8da', '#b5844e'
BAND_A, BAND_B = '#eef4f9', '#f8fbfd'
FILL, FILL_HI, MUTED = '#dbe7f3', '#bcd3e8', '#eef3f8'

fig, ax = plt.subplots(figsize=(11.6, 7.4))
ax.set_xlim(0, 116); ax.set_ylim(0, 80); ax.axis('off')

def band(y, h, colour, title, sub):
    ax.add_patch(FancyBboxPatch((1.5, y), 113, h, boxstyle='round,pad=0.6',
                                fc=colour, ec='#b7cde1', lw=1.1, zorder=0))
    t = ax.text(4.5, y + h - 3.6, title, fontsize=12.5, style='italic',
                weight='bold', color=INK, va='center', zorder=3)
    fig.canvas.draw()
    bb = t.get_window_extent().transformed(ax.transData.inverted())
    ax.text(bb.x1 + 1.0, y + h - 3.6, sub, fontsize=12.5, weight='bold',
            color=INK, va='center', zorder=3)
    ax.plot([4.5, 111], [y + h - 6.2, y + h - 6.2], color='#b7cde1', lw=0.9, zorder=1)

def box(x, y, w, h, lines, fill=FILL, edge='#7fa3c4', lw=1.0, fs=9.0):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.35',
                                fc=fill, ec=edge, lw=lw, zorder=2))
    ax.text(x + w/2, y + h/2, '\n'.join(lines), ha='center', va='center',
            fontsize=fs, color=INK, linespacing=1.55, zorder=3)
    return (x, y, w, h)

def arrow(a, b, rad=0.0, colour=ACC, lw=1.4):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle='-|>', mutation_scale=12,
                                 color=colour, lw=lw, shrinkA=2, shrinkB=2,
                                 connectionstyle=f'arc3,rad={rad}', zorder=4))

def chain(src, dst):
    xs, ys, ws, hs = src; xd, yd, wd, hd = dst
    if xd > xs: arrow((xs+ws, ys+hs/2), (xd, yd+hd/2))
    else:       arrow((xs, ys+hs/2), (xd+wd, yd+hd/2))

# ============================================================== inherited
band(61.0, 17.5, BAND_A, 'Inherited:', ' corpus, panel protocol, human subset')
yA = 62.0
i1 = box(  5.0, yA, 18.5,  9.5, ['807 UCF-Crime', 'videos + UCA', 'reference annot.'])
i2 = box( 27.0, yA, 18.5,  9.5, ['3 source LLMs', '\u00d7 8 prompting', 'strategies'])
i3 = box( 49.0, yA, 18.5,  9.5, ['19,361 reports', '21.8M words', 'fully crossed'])
i4 = box( 71.0, yA, 19.5,  9.5, ['cross-judging panel', '2 of 3 judges,', 'no self-evaluation'])
i5 = box( 94.0, yA, 17.0,  9.5, ['38,722', 'per-judge', 'H1\u2013H6 verdicts'])
for s, d in [(i1,i2),(i2,i3),(i3,i4),(i4,i5)]:
    chain(s, d)

# ------------------------------------- human subset, in the gap between bands
hs = box(48.0, 50.5, 26.0, 6.6, ['130-report subset, 2 human raters'], fill=MUTED)
ax.add_patch(FancyArrowPatch((i3[0]+i3[2]/2, i3[1]), (hs[0]+3.0, hs[1]+hs[3]),
             arrowstyle='-|>', mutation_scale=12, color=ACC, lw=1.4,
             shrinkA=2, shrinkB=3, connectionstyle='arc3,rad=-0.32', zorder=4))
ax.text(40.0, 58.0, 'stratified', fontsize=8.3, style='italic', color=ACC, zorder=5)

# ============================================================== this paper
band(2.0, 46.0, BAND_B, 'This paper:', ' labels, splits, detectors, diagnosis')
yB = 27.0

def step(n, bx):
    ax.text(bx[0] + bx[2]/2, bx[1] + bx[3] + 2.4, str(n), fontsize=10.5,
            weight='bold', color='white', ha='center', va='center', zorder=6,
            bbox=dict(boxstyle='circle,pad=0.3', fc=ACC, ec='none'))

s1 = box(88.0, yB, 23.0, 12.0,
         ['two-judge conjunction', '+ per-type reliability tiers',
          'H1 low ($\\kappa\\,{=}\\,0.15$), excluded'], fill=FILL_HI, edge=ACC, lw=1.6)
s2 = box(58.0, yB, 27.0, 12.0, [''], fill=FILL_HI, edge=ACC, lw=1.6)
ax.text(71.5, yB + 10.4, '4 video-disjoint splits', fontsize=8.8,
        ha='center', color=INK, zorder=3)

def split_panel(x0, y0, w, h, test_cols=None, test_rows=None, title=''):
    """Miniature 3-row grid: rows are source LLMs, columns video-by-strategy."""
    nc, nr = 8, 3
    cw, ch = w / nc, h / nr
    for r in range(nr):
        for c in range(nc):
            hot = (test_cols is not None and c in test_cols) or \
                  (test_rows is not None and r in test_rows)
            ax.add_patch(plt.Rectangle((x0 + c*cw, y0 + (nr-1-r)*ch), cw, ch,
                                       fc=('#2d5d8a' if hot else '#ffffff'),
                                       ec='#8fb0cc', lw=0.35, zorder=3))
    ax.text(x0 + w/2, y0 + h + 0.7, title, fontsize=6.6, ha='center',
            color=INK, zorder=4)

split_panel(59.6, yB + 3.6, 7.2, 4.2, test_cols={6, 7},        title='standard')
split_panel(68.4, yB + 3.6, 7.2, 4.2, test_rows={2},           title='held-out source')
split_panel(77.2, yB + 3.6, 7.2, 4.2, test_cols={1, 4, 6},     title='held-out strategy')
ax.add_patch(plt.Rectangle((60.2, yB + 1.3), 1.5, 1.1, fc='#ffffff',
                           ec='#8fb0cc', lw=0.4, zorder=3))
ax.text(62.1, yB + 1.85, 'train', fontsize=6.4, va='center', color=INK, zorder=4)
ax.add_patch(plt.Rectangle((68.0, yB + 1.3), 1.5, 1.1, fc='#2d5d8a',
                           ec='#8fb0cc', lw=0.4, zorder=3))
ax.text(69.9, yB + 1.85, 'test', fontsize=6.4, va='center', color=INK, zorder=4)
ax.text(76.5, yB + 1.85, '+ strict variant', fontsize=6.4, va='center',
        color=INK, style='italic', zorder=4)
s3 = box(26.0, yB, 30.0, 12.0,
         ['detectors', 'surface · TF-IDF · embedding',
          'reference-grounded · entailment',
          'fine-tuned encoders'], fill=FILL_HI, edge=ACC, lw=1.6)
s4 = box( 5.0, yB, 17.5, 12.0,
         ['per-type AUC', 'macro, macro-hq', 'bootstrap CI', 'on the gap'])
for n, b in [(1,s1),(2,s2),(3,s3),(4,s4)]:
    step(n, b)

arrow((i5[0]+i5[2]/2, i5[1]), (s1[0]+s1[2]*0.70, s1[1]+s1[3]))
arrow((hs[0]+hs[2], hs[1]+hs[3]/2), (s1[0]+s1[2]*0.25, s1[1]+s1[3]), rad=0.30)
ax.text(78.5, 45.5, 'calibrates', fontsize=8.3, style='italic', color=ACC, zorder=5)
chain(s1, s2); chain(s2, s3); chain(s3, s4)

# ------------------------------------------------- the control, feeding step 3
ctl = box(24.5, 15.8, 33.0, 8.4,
          ['base-rate control', 'source identity only, no report text',
           'how much of the signal is the source LLM, not the claim?'],
          fill=TAN, edge=TAN_E, lw=1.4, fs=8.4)
# the control is a parallel path: it takes the same split and is scored
# alongside the detectors, rather than feeding them
arrow((s2[0] + 2.0, s2[1]), (ctl[0] + ctl[2], ctl[1] + ctl[3]/2),
      rad=-0.25, colour=TAN_E, lw=1.3)
arrow((ctl[0], ctl[1] + ctl[3]/2), (s4[0] + s4[2]/2, s4[1]),
      rad=-0.25, colour=TAN_E, lw=1.3)


# ------------------------------------------------------------------ findings
ax.add_patch(FancyBboxPatch((5.0, 3.2), 106.0, 10.6, boxstyle='round,pad=0.4',
                            fc='white', ec='#c9d9e8', lw=1.0, zorder=2))
ax.text(8.5, 11.6, 'Findings', fontsize=9.6, weight='bold', color=INK, zorder=3)
rows = [
 ('RQ1\u2013RQ2', 'detection degrades under source shift; the fabrication labels lose two to three times the AUC'),
 ('RQ3',        'base rate alone, with no report text, gives 70\u201373% of the fabrication signal against 13% on omission'),
 ('RQ4',        'grounding recovers binary detection; entailment does not close the omission gap'),
 ('RQ5',        'panel labels track report length; each source LLM gets its own pair of judges'),
]
for k, (tag, txt) in enumerate(rows):
    yy = 9.6 - k * 2.0
    ax.text(9.5, yy, tag, fontsize=8.3, weight='bold', color=ACC, zorder=3)
    ax.text(21.0, yy, txt, fontsize=8.3, color=INK, zorder=3)
arrow((s4[0]+s4[2]/2, s4[1]), (s4[0]+s4[2]/2, 13.8))

plt.tight_layout(pad=0.25)
plt.savefig('fig_pipeline.pdf', bbox_inches='tight')
plt.savefig('fig_pipeline.png', bbox_inches='tight', dpi=260)
print('wrote fig_pipeline.pdf and .png')
