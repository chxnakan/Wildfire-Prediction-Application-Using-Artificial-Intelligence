"""Render the requested poster infographic using actual frozen 72-hour evidence.

Run: python create_infographic.py
Dependencies: matplotlib, numpy, pandas, scikit-learn.
The adjacent plot_data folder makes this script independent of the project.
"""
from pathlib import Path
import hashlib
import json

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from matplotlib.transforms import Bbox
import numpy as np
import pandas as pd
from sklearn.metrics import (confusion_matrix, roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score)

OUT = Path(__file__).resolve().parent
DATA = OUT / 'plot_data'
BG, NAVY = '#C6D8FA', '#343B7C'
BLUE, ORANGE, GREEN = '#9ABFE7', '#BD713C', '#DCEEE0'
RED, INK, MUTED, GRID = '#F5DCD7', '#243447', '#536477', '#E8ECF1'


def load_evidence():
    evidence=json.loads((DATA/'evidence.json').read_text())
    file=DATA/'prediction_scores.csv.gz'
    assert hashlib.sha256(file.read_bytes()).hexdigest()==evidence['prepared_data_sha256'][file.name]
    assert evidence['horizon_days']==3
    scores=pd.read_csv(file,parse_dates=['date'])
    y=scores.target.to_numpy(dtype=int)
    p=scores.probability.to_numpy(dtype=float)
    assert np.isin(y,[0,1]).all() and np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
    assert not scores.duplicated(['date','lat_cell','lon_cell']).any()
    assert str(scores.date.min().date())==evidence['split']['test_start']
    assert str(scores.date.max().date())==evidence['split']['test_end']
    expected=evidence['saved_model_metrics']['classification']
    threshold=float(expected['threshold'])
    tn,fp,fn,tp=map(int,confusion_matrix(y,p>=threshold,labels=[0,1]).ravel())
    m={'roc_auc':float(roc_auc_score(y,p)), 'recall':tp/(tp+fn),
       'precision':tp/(tp+fp), 'average_precision':float(average_precision_score(y,p)),
       'true_positive':tp, 'false_positive':fp, 'false_negative':fn,'true_negative':tn}
    for key,value in m.items():
        assert np.isclose(value,expected[key],rtol=1e-10,atol=1e-12),key
    m.update(prevalence=float(y.mean()),threshold=threshold,samples=len(y),
        model_version=evidence['model_version'],horizon_days=3,split=evidence['split'],
        source='Actual saved chronological holdout, not illustrative data')
    return y,p,m


def rounded(fig,x,y,w,h,fill='white',radius=.014):
    fig.add_artist(FancyBboxPatch((x,y),w,h,
        boxstyle=f'round,pad=0,rounding_size={radius}',
        transform=fig.transFigure,facecolor=fill,edgecolor='none',zorder=-2))


def graph_axes(fig,box,xlabel,ylabel):
    ax=fig.add_axes(box,facecolor='none')
    ax.set(xlim=(0,1),ylim=(0,1),xlabel=xlabel,ylabel=ylabel)
    ax.xaxis.set_major_locator(MultipleLocator(.2))
    ax.yaxis.set_major_locator(MultipleLocator(.2))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.grid(color=GRID,linewidth=.8)
    ax.set_axisbelow(True)
    return ax


def export_sections(fig):
    """Export the original vector artists individually, with no resampling."""
    dest=OUT/'separated'
    dest.mkdir(exist_ok=True)
    regions={
        '01_overall_performance':(.04,.585,.445,.275),
        '02_confusion_matrix':(.515,.585,.445,.275),
        '03_roc_curve':(.04,.270,.445,.290),
        '04_precision_recall_curve':(.515,.270,.445,.290),
        '05_key_results':(.04,.133,.920,.107),
        '06_evaluation_information':(.04,.055,.920,.057),
        '07_roc_auc_card':(.04,.133,.290,.107),
        '08_recall_card':(.355,.133,.290,.107),
        '09_precision_card':(.670,.133,.290,.107),
    }
    objects=list(fig.axes)+list(fig.texts)+list(fig.artists)
    original={obj:obj.get_visible() for obj in objects}
    width,height=fig.get_size_inches()
    exports=[]
    for name,(x,y,w,h) in regions.items():
        def contains(px,py):
            return x <= px <= x+w and y <= py <= y+h
        for obj in objects:
            if obj in fig.axes:
                b=obj.get_position()
                selected=contains(b.x0+b.width/2,b.y0+b.height/2)
            elif obj in fig.texts:
                selected=contains(*obj.get_position())
            else:
                selected=contains(obj.get_x()+obj.get_width()/2,obj.get_y()+obj.get_height()/2)
            obj.set_visible(original[obj] and selected)
        margin=.008
        bounds=Bbox.from_bounds((x-margin)*width,(y-margin)*height,
                                (w+2*margin)*width,(h+2*margin)*height)
        fig.savefig(dest/f'{name}.png',dpi=300,facecolor=BG,bbox_inches=bounds,pad_inches=0)
        fig.savefig(dest/f'{name}.svg',facecolor=BG,bbox_inches=bounds,pad_inches=0,
                    metadata={'Date':None})
        exports.append({'name':name,'figure_bounds':[x,y,w,h],
                        'size_inches':[(w+2*margin)*width,(h+2*margin)*height], 'dpi':300})
    for obj,visible in original.items():obj.set_visible(visible)
    (dest/'export_manifest.json').write_text(json.dumps(exports,indent=2)+'\n')


def main():
    y,p,m=load_evidence()
    threshold=m['threshold']
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':13,
        'text.color':INK,'axes.labelcolor':INK,'axes.labelsize':14,
        'axes.labelpad':9,'axes.edgecolor':'#ABB7C5','axes.linewidth':.8,
        'axes.spines.top':False,'axes.spines.right':False,
        'xtick.labelsize':12,'ytick.labelsize':12,'xtick.color':MUTED,'ytick.color':MUTED,
        'legend.fontsize':11,'legend.frameon':False,'svg.fonttype':'none',
        'svg.hashsalt':'wildfire-72h-infographic','figure.dpi':100})
    fig=plt.figure(figsize=(16,17),facecolor=BG)
    fig.text(.045,.972,'Wildfire Prediction Web Application Using Artificial Intelligence',
        fontsize=18,color=NAVY,weight='semibold',va='top')
    fig.text(.045,.931,'Model Performance & Validation',fontsize=32,color=NAVY,
        weight='bold',va='top')
    fig.text(.045,.892,f'Actual 72-hour evaluation  •  {m["samples"]:,} grid-day windows  •  Thailand',
        fontsize=14,color=INK,va='top')

    left,right,width=.04,.515,.445
    for x in [left,right]:
        rounded(fig,x,.585,width,.275)
        rounded(fig,x,.270,width,.290)
    fig.text(.062,.836,'A. Overall model performance',fontsize=19,color=NAVY,weight='bold')
    fig.text(.537,.836,'B. Confusion matrix',fontsize=19,color=NAVY,weight='bold')
    fig.text(.062,.536,'C. Receiver operating characteristic',fontsize=17,color=NAVY,weight='bold')
    fig.text(.537,.536,'D. Precision–recall curve',fontsize=19,color=NAVY,weight='bold')

    # Threshold-independent AUC and threshold-specific recall/precision.
    ax=fig.add_axes([.095,.660,.350,.145],facecolor='none')
    values=[m['roc_auc'],m['recall'],m['precision']]
    bars=ax.bar(['ROC-AUC','Recall','Precision'],values,width=.57,color=[NAVY,BLUE,ORANGE],zorder=3)
    for bar,label in zip(bars,[f'{values[0]:.3f}',f'{values[1]:.1%}',f'{values[2]:.1%}']):
        ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+.035,label,
            ha='center',fontsize=17,weight='bold',color=NAVY)
    ax.set_ylim(0,1)
    ax.set_ylabel('Metric value')
    ax.yaxis.set_major_locator(MultipleLocator(.2))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.grid(axis='y',color=GRID,linewidth=.8)
    ax.set_axisbelow(True)
    fig.text(.062,.611,'ROC-AUC: across thresholds; not percent accuracy.\n'
        f'Recall and precision: alert threshold = {threshold:.2f}.',fontsize=11.5,color=MUTED,
        linespacing=1.6)

    # Real confusion counts; rows actual positive/negative, columns predicted positive/negative.
    ax=fig.add_axes([.680,.653,.245,.137],facecolor='none')
    counts=[[m['true_positive'],m['false_negative']],[m['false_positive'],m['true_negative']]]
    labels=[['True positive (TP)','False negative (FN)'],['False positive (FP)','True negative (TN)']]
    for row in range(2):
        for col in range(2):
            ax.add_patch(Rectangle((col+.015,row+.015),.97,.97,
                facecolor=GREEN if row==col else RED,edgecolor='white',linewidth=1.5))
            ax.text(col+.5,row+.43,f'{counts[row][col]:,}',ha='center',va='center',
                color=NAVY,fontsize=24,weight='bold')
            ax.text(col+.5,row+.72,labels[row][col],ha='center',va='center',color=INK,fontsize=10.5)
    ax.set(xlim=(0,2),ylim=(2,0))
    ax.set_xticks([.5,1.5],['Hotspot\npredicted','No hotspot\npredicted'])
    ax.set_yticks([.5,1.5],['Hotspot\nrecorded','No hotspot\nrecorded'])
    ax.xaxis.tick_top()
    ax.tick_params(length=0,labelsize=11,pad=9)
    for spine in ax.spines.values():spine.set_visible(False)
    fig.text(.537,.611,'Actual held-out counts. Green = correct; coral = incorrect.\n'
        'Rows: recorded outcome. Columns: model prediction.',fontsize=11.5,color=MUTED,linespacing=1.6)

    # Actual empirical ROC, with a light-blue area and chance diagonal.
    ax=graph_axes(fig,[.095,.340,.350,.168],'False positive rate','True positive rate')
    fpr,tpr,_=roc_curve(y,p)
    ax.fill_between(fpr,0,tpr,color=BLUE,alpha=.20,linewidth=0)
    ax.plot(fpr,tpr,color=NAVY,linewidth=2.7,label=f'ROC-AUC = {m["roc_auc"]:.3f}')
    ax.plot([0,1],[0,1],color='#8793A0',linewidth=1.5,linestyle='--',label='Random classification')
    false_rate=m['false_positive']/(m['false_positive']+m['true_negative'])
    ax.scatter([false_rate],[m['recall']],s=48,color=ORANGE,zorder=5,label=f'Threshold = {threshold:.2f}')
    ax.legend(loc='lower right',fontsize=10.5)
    fig.text(.062,.290,'Computed from every held-out probability and true label.',fontsize=11.5,color=MUTED)

    # Actual empirical PR and prevalence baseline; AP is not invented.
    ax=graph_axes(fig,[.570,.340,.350,.168],'Recall','Precision')
    precision,recall,_=precision_recall_curve(y,p)
    ax.step(recall,precision,where='post',color=ORANGE,linewidth=2.5,
        label=f'Average precision = {m["average_precision"]:.3f}')
    ax.axhline(m['prevalence'],color='#8793A0',linestyle='--',linewidth=1.5,
        label=f'Event prevalence = {m["prevalence"]:.2%}')
    ax.scatter([m['recall']],[m['precision']],s=48,color=NAVY,zorder=5,label=f'Threshold = {threshold:.2f}')
    ax.legend(loc='upper right',fontsize=10.5)
    fig.text(.537,.290,f'{m["true_positive"]+m["false_negative"]:,} positive windows out of {m["samples"]:,} evaluated.',
        fontsize=11.5,color=MUTED)

    # Key results: three clearly differentiated, readable metric cards.
    cards=[(.04,'ROC-AUC',f'{m["roc_auc"]:.3f}','Model discrimination\nNot classification accuracy',
        '#E7ECFA',NAVY),
        (.355,'RECALL',f'{m["recall"]:.1%}','Of hotspot-positive windows,\nthis share was detected.',
        '#E4F0FC','#39668D'),
        (.670,'PRECISION',f'{m["precision"]:.1%}','Of flagged windows, this share\nhad a recorded hotspot.',
        '#FAE9DB',ORANGE)]
    for x,label,value,caption,fill,color in cards:
        rounded(fig,x,.133,.290,.107,fill=fill)
        fig.text(x+.020,.220,label,fontsize=12,color=NAVY,weight='bold')
        fig.text(x+.020,.182,value,fontsize=31,color=color,weight='bold')
        fig.text(x+.020,.145,caption,fontsize=11.5,color=INK,linespacing=1.4)

    rounded(fig,.04,.055,.920,.057)
    fields=[(.062,'FORECAST HORIZON','72 hours (days 1–3)'),
        (.360,'HISTORICAL TEST PERIOD','6 Nov 2025 – 14 Jan 2026'),
        (.745,'ALERT THRESHOLD',f'{threshold:.2f}')]
    for x,label,value in fields:
        fig.text(x,.092,label,fontsize=10.5,color=MUTED,weight='bold')
        fig.text(x,.070,value,fontsize=14,color=NAVY,weight='semibold')
    fig.text(.045,.023,'Satellite hotspots are not confirmed wildfires. Test windows overlap; they are not separate fire incidents.',
        fontsize=10.5,color=NAVY)

    fig.savefig(OUT/'model-performance-validation.png',dpi=300,facecolor=BG)
    fig.savefig(OUT/'model-performance-validation.svg',facecolor=BG,metadata={'Date':None})
    export_sections(fig)
    plt.close(fig)
    m.update(output_pixels=[4800,5100],dpi=300,illustrative_data_used=False)
    (OUT/'verified_metrics.json').write_text(json.dumps(m,indent=2)+'\n')
    print(json.dumps(m,indent=2))


if __name__=='__main__':main()
