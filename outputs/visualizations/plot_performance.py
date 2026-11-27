"""Plot the saved historical holdout without changing the app or model."""
from pathlib import Path
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
metrics = json.loads((ROOT / 'outputs/evaluation/metrics.json').read_text())
c = metrics['model']['classification']
data = pd.read_csv(ROOT / 'outputs' / metrics['evidence_directory'] / 'holdout_predictions.csv.gz')
assert metrics['horizon_days'] == 3
predicted = data.probability >= c['threshold']
actual = data.target.astype(bool)
assert int((predicted & actual).sum()) == c['true_positive']
assert int((predicted & ~actual).sum()) == c['false_positive']
assert int((~predicted & actual).sum()) == c['false_negative']
order = np.argsort(-data.probability.to_numpy(), kind='stable')
scores = data.probability.to_numpy()[order]
targets = actual.to_numpy()[order]
ends = np.r_[np.flatnonzero(np.diff(scores)), len(scores)-1]
true_counts = np.cumsum(targets)[ends]
false_counts = 1 + ends - true_counts
tpr = np.r_[0, true_counts / actual.sum()]
fpr = np.r_[0, false_counts / (~actual).sum()]
auc = np.sum(np.diff(fpr) * (tpr[1:] + tpr[:-1]) / 2)
assert np.isclose(auc, c['roc_auc'], rtol=1e-10)

ink, muted, teal, coral = '#173849', '#526b79', '#087f8c', '#e9a091'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
    'text.color': ink, 'axes.labelcolor': ink, 'xtick.color': muted,
    'ytick.color': muted, 'axes.spines.top': False, 'axes.spines.right': False})
fig = plt.figure(figsize=(13.6, 7.4), facecolor='#f7fafb')
fig.text(.065, .925, 'Three-day hotspot forecast performance', fontsize=24, weight='bold')
fig.text(.065, .875, 'THAILAND  |  259,860 test windows  |  6 Nov 2025 – 14 Jan 2026',
         fontsize=11, color=muted)

ax = fig.add_axes([.075, .24, .36, .51], facecolor='#f7fafb')
ax.plot(fpr, tpr, color=teal, linewidth=2.7, label=f'Model · ROC-AUC {c["roc_auc"]:.3f}')
ax.plot([0,1], [0,1], color='#8f9ca4', linewidth=1.3, linestyle='--', label='Random ranking · AUC 0.500')
ax.scatter([c['false_positive_rate']], [c['recall']], s=65, color=ink, zorder=4)
ax.annotate('50% alert threshold', xy=(c['false_positive_rate'], c['recall']),
            xytext=(.27,.26), arrowprops={'arrowstyle':'-', 'color':ink}, fontsize=10)
ax.set(xlim=(0,1), ylim=(0,1), xlabel='False-positive rate', ylabel='Event capture (recall)')
ax.set_title('Risk ranking across alert thresholds', loc='left', pad=20, fontsize=13, weight='bold')
ax.xaxis.set_major_formatter(PercentFormatter(1))
ax.yaxis.set_major_formatter(PercentFormatter(1))
ax.grid(color='#dce5e9', linewidth=.65)
ax.set_axisbelow(True)
for spine in ax.spines.values():
    spine.set_color('#c5d3da')
ax.legend(loc='lower right', frameon=False, fontsize=9)

fig.text(.51,.79,'Results at the 50% alert threshold',fontsize=13,weight='bold')
bars = [
    (.59, 'Of windows that contained a hotspot', c['recall'],
     'Caught', 'Missed', f'{c["true_positive"]:,} caught / {c["actual_positive_windows"]:,} hotspot-positive windows'),
    (.35, 'Of windows flagged by the model', c['precision'],
     'Hotspot recorded', 'No hotspot recorded',
     f'{c["true_positive"]:,} with a hotspot / {c["true_positive"] + c["false_positive"]:,} flagged windows')]
for bottom, title, share, left_label, right_label, detail in bars:
    bx = fig.add_axes([.51, bottom, .43, .075], facecolor='#f7fafb')
    bx.barh([0],[share],color=teal,height=.7)
    bx.barh([0],[1-share],left=[share],color=coral,height=.7)
    bx.text(share/2,0,f'{share:.1%}',ha='center',va='center',color='white',fontsize=15,weight='bold')
    bx.text(share+(1-share)/2,0,f'{1-share:.1%}',ha='center',va='center',color=ink,fontsize=15,weight='bold')
    bx.set(xlim=(0,1),ylim=(-.5,.5))
    bx.axis('off')
    fig.text(.51,bottom+.095,title,fontsize=12,weight='bold')
    fig.text(.51,bottom-.021,left_label,fontsize=10,color=teal)
    fig.text(.94,bottom-.021,right_label,fontsize=10,color=ink,ha='right')
    fig.text(.51,bottom-.06,detail,fontsize=10,color=muted)

fig.text(.065,.125,'A hotspot is a satellite detection, not a confirmed wildfire. Test windows overlap; they are not separate fire incidents.',
         fontsize=10,color=muted)
fig.text(.065,.087,'Current saved model: 98bfd2a97aab  •  Newly supplied downloads are not included in these results.',fontsize=10,color=muted)
fig.savefig(OUT/'three-day-performance.png',dpi=180,facecolor=fig.get_facecolor())
plt.close(fig)
print(OUT/'three-day-performance.png')
