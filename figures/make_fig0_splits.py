import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
plt.rcParams.update({'font.size':7.5,'savefig.bbox':'tight'})

GEN=['Claude','GPT','Gemini']; NV=9
TR='#cfd8e3'; TE='#4a6fa5'; ED='#6b7a8d'

fig,axes=plt.subplots(1,3,figsize=(6.6,1.85))
panels=[('(a) Standard split','video'),
        ('(b) Held-out generator','gen'),
        ('(c) Held-out strategy','tech')]

for ax,(title,mode) in zip(axes,panels):
    for r,g in enumerate(GEN):
        for c in range(NV):
            if mode=='video':   test = c>=NV-3
            elif mode=='gen':   test = (g=='Gemini')
            else:               test = c%3==2
            ax.add_patch(Rectangle((c,-r),1,1,facecolor=TE if test else TR,
                                   edgecolor=ED,linewidth=.4))
    if mode=='video':
        ax.plot([NV-3,NV-3],[-3,1],color='k',lw=1.6)
    elif mode=='gen':
        ax.plot([0,NV],[-1.99,-1.99],color='k',lw=1.6)
    ax.set_xlim(-.2,NV+.2); ax.set_ylim(-3.25,1.5)
    ax.set_yticks([.5,-.5,-1.5]); ax.set_yticklabels(GEN,fontsize=7)
    ax.set_xticks([]); ax.set_title(title,fontsize=8,pad=3)
    ax.set_frame_on(False); ax.tick_params(length=0)
    ax.text(NV/2,1.15,'videos $\\times$ strategies $\\rightarrow$',
            ha='center',fontsize=6.5,color='0.35')

h=[Rectangle((0,0),1,1,facecolor=TR,edgecolor=ED),
   Rectangle((0,0),1,1,facecolor=TE,edgecolor=ED)]
fig.legend(h,['train','test'],ncol=2,frameon=False,fontsize=7.5,
           loc='lower center',bbox_to_anchor=(.5,-.10))
fig.savefig('fig0_splits.pdf')
print('ok')
