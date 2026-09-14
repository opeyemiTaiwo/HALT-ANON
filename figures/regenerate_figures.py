import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, numpy as np, pandas as pd
from matplotlib.patches import Rectangle
plt.rcParams.update({'font.size':7.5,'savefig.bbox':'tight'})
R='/home/claude/icl/iclr_benchmark'
H=['H1','H2','H3','H4','H5','H6']
AX={'fabrication':['H1','H5','H6'],'omission':['H3'],'distortion':['H2','H4']}

# fig0 — splits schematic
GEN=['Claude','GPT','Gemini']; NV=9; TR='#cfd8e3'; TE='#4a6fa5'; ED='#6b7a8d'
fig,axes=plt.subplots(1,3,figsize=(6.6,1.85))
for ax,(title,mode) in zip(axes,[('(a) Standard split','video'),
                                 ('(b) Held-out source LLM','gen'),
                                 ('(c) Held-out strategy','tech')]):
    for r,g in enumerate(GEN):
        for c in range(NV):
            test = c>=NV-3 if mode=='video' else (g=='Gemini' if mode=='gen' else c%3==2)
            ax.add_patch(Rectangle((c,-r),1,1,facecolor=TE if test else TR,edgecolor=ED,linewidth=.4))
    if mode=='video': ax.plot([NV-3,NV-3],[-3,1],color='k',lw=1.6)
    elif mode=='gen': ax.plot([0,NV],[-1.99,-1.99],color='k',lw=1.6)
    ax.set_xlim(-.2,NV+.2); ax.set_ylim(-3.25,1.5)
    ax.set_yticks([.5,-.5,-1.5]); ax.set_yticklabels(GEN,fontsize=7)
    ax.set_xticks([]); ax.set_title(title,fontsize=8,pad=3)
    ax.set_frame_on(False); ax.tick_params(length=0)
    ax.text(NV/2,1.15,'videos $\\times$ strategies $\\rightarrow$',ha='center',fontsize=6.5,color='0.35')
fig.legend([Rectangle((0,0),1,1,facecolor=TR,edgecolor=ED),
            Rectangle((0,0),1,1,facecolor=TE,edgecolor=ED)],['train','test'],
           ncol=2,frameon=False,fontsize=7.5,loc='lower center',bbox_to_anchor=(.5,-.10))
fig.savefig('fig0_splits.pdf'); plt.close()

# fig1 — the inversion
base=pd.read_csv(f'{R}/baseline_results.csv'); logo=pd.read_csv(f'{R}/logo_results.csv')
ci=pd.read_csv(f'{R}/logo_auc_ci.csv')
rnd=base[(base.features=='tfidf')&(base.split=='random')].set_index('target').auc
lg=logo[logo.features=='tfidf'].groupby('target').auc.mean()
cim=ci.groupby('target')[['ci_low','ci_high']].mean()
x=np.arange(6); w=0.38
fig,ax=plt.subplots(figsize=(5.5,2.6))
ax.bar(x-w/2,[rnd[h] for h in H],w,label='same source LLM')
ax.bar(x+w/2,[lg[h] for h in H],w,label='held-out source LLM',
       yerr=[[lg[h]-cim.ci_low[h] for h in H],[cim.ci_high[h]-lg[h] for h in H]],
       capsize=2,ecolor='0.3')
ax.axhline(0.5,ls=':',c='gray',lw=.8)
ax.set_xticks(x); ax.set_xticklabels(['H1\nScene Fab','H2\nMisclass','H3\nMissed',
                                      'H4\nSeverity','H5\nEntity Fab','H6\nPhantom'])
ax.set_ylabel('AUC'); ax.set_ylim(0.45,0.95); ax.legend(frameon=False,ncol=2)
fig.savefig('fig1_inversion.pdf'); plt.close()

# fig2 — feature sets
piv=logo.pivot_table(index='features',columns='target',values='auc')
order=[f for f in ['tfidf','embed','grounded','grounded+surface'] if f in piv.index]
fig,ax=plt.subplots(figsize=(5.5,2.6))
for a,cs in AX.items(): ax.plot(order,[piv.loc[f,cs].mean() for f in order],marker='o',label=a)
ax.plot(order,[piv.loc[f,'any_hallucination'] for f in order],marker='s',ls='--',c='k',label='ANY')
ax.axhline(0.5,ls=':',c='gray',lw=.8)
ax.set_ylabel('AUC (held-out source LLM)'); ax.legend(frameon=False,fontsize=7)
plt.xticks(rotation=15); fig.savefig('fig2_featuresets.pdf'); plt.close()
print('figures regenerated')
