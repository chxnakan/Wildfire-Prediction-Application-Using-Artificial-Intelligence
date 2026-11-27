"""Reproduce poster figures from real, frozen three-day holdout evidence.

Prepare from the project, then plot:
    python generate_figures.py --project-root /path/to/project
Replot the supplied, portable evidence bundle:
    python generate_figures.py

Dependencies: Matplotlib, Pandas, NumPy, Scikit-learn. No network requests.
The dashboard, model, source observations and evaluation artifacts are read-only.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform

import matplotlib
matplotlib.use('Agg')
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, MultipleLocator, StrMethodFormatter
import numpy as np
import pandas as pd
import sklearn
from sklearn.calibration import calibration_curve
from sklearn.metrics import (average_precision_score, brier_score_loss,
    confusion_matrix, precision_recall_curve, roc_auc_score, roc_curve)

NAVY = '#173B5E'
BLUE = '#8EBBDA'
ORANGE = '#B6682E'
LIGHT_ORANGE = '#EBC5A4'
CORAL = '#CB7766'
GRAY = '#747D86'
GRID = '#E6E9ED'
INK = '#243746'


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def inside_ring(x, y, ring):
    inside = False
    xj, yj = ring[-1]
    for xi, yi in ring:
        if (yi > y) != (yj > y) and x < (xj-xi)*(y-yi)/(yj-yi)+xi:
            inside = not inside
        xj, yj = xi, yi
    return inside


def country_membership(cells, grid, boundary):
    """Match the project's grid-centre policy, including polygon holes."""
    polygons = []
    for feature in boundary['features']:
        geom = feature['geometry']
        parts = [geom['coordinates']] if geom['type'] == 'Polygon' else geom['coordinates']
        for rings in parts:
            xs, ys = zip(*rings[0])
            polygons.append(((min(xs), min(ys), max(xs), max(ys)), rings))
    keep = []
    for lat_cell, lon_cell in cells.itertuples(index=False, name=None):
        x, y = lon_cell * grid, lat_cell * grid
        belongs = any(w <= x <= e and s <= y <= n
            and inside_ring(x, y, rings[0])
            and not any(inside_ring(x, y, ring) for ring in rings[1:])
            for (w, s, e, n), rings in polygons)
        keep.append(belongs)
    return np.asarray(keep)


def prepare(project, data_dir):
    data_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = project / 'outputs/evaluation/metrics.json'
    dataset_path = project / 'outputs/metadata/dataset_info.json'
    coordinate_path = project / 'outputs/features/observation_coordinates.csv.gz'
    boundary_path = project / 'frontend/data/thailand-provinces.geojson'
    metrics = json.loads(metrics_path.read_text())
    dataset = json.loads(dataset_path.read_text())
    if metrics.get('horizon_days') != 3:
        raise ValueError('Expected a documented three-day evaluation; cannot relabel a seven-day test.')
    if dataset['model_version'] != Path(metrics['evidence_directory']).name:
        raise ValueError('Dataset and evaluation refer to different models.')
    prediction_path = project / 'outputs' / metrics['evidence_directory'] / 'holdout_predictions.csv.gz'
    columns = ['date','lat_cell','lon_cell','target','probability','days_until_detection']
    predictions = pd.read_csv(prediction_path, usecols=columns)
    predictions.to_csv(data_dir / 'prediction_scores.csv.gz', index=False,
                       compression={'method': 'gzip', 'mtime': 0})

    obs = pd.read_csv(coordinate_path, parse_dates=['date'])
    # Saved coordinate evidence is already deduplicated by satellite, minute and pixel.
    keys = ['satellite','observed_at','latitude','longitude']
    if obs.duplicated(keys).any():
        raise ValueError('Observation evidence contains duplicate detection keys.')
    start = obs.date.min()
    end = pd.Timestamp(metrics['split']['test_end']) + pd.Timedelta(days=metrics['horizon_days'])
    obs = obs[obs.date.between(start, end)].copy()
    calendar = pd.date_range(start, end, freq='D')
    regional = obs.groupby('date').size().reindex(calendar, fill_value=0)
    grid = float(dataset['grid_size'])
    obs['lat_cell'] = np.rint(obs.latitude / grid).astype(int)
    obs['lon_cell'] = np.rint(obs.longitude / grid).astype(int)
    cells = obs[['lat_cell','lon_cell']].drop_duplicates()
    boundary = json.loads(boundary_path.read_text())
    valid_cells = pd.MultiIndex.from_frame(cells[country_membership(cells, grid, boundary)])
    mask = pd.MultiIndex.from_frame(obs[['lat_cell','lon_cell']]).isin(valid_cells)
    eligible = obs[mask & ~obs.hotspot_type.isin([1,2,3])]
    count = eligible.groupby('date').size().reindex(calendar, fill_value=0).astype(float)
    active = eligible.drop_duplicates(['date','lat_cell','lon_cell']).groupby('date').size().reindex(calendar, fill_value=0).astype(float)
    # No regional record means unknown coverage, not an observed zero.
    count = count.where(regional > 0)
    active = active.where(regional > 0)
    daily = pd.DataFrame({'date': calendar, 'hotspots': count.to_numpy(),
        'active_cells': active.to_numpy(), 'regional_records': regional.to_numpy()})
    daily.to_csv(data_dir / 'daily_activity.csv', index=False)
    evidence = {
        'model_version': dataset['model_version'], 'horizon_days': metrics['horizon_days'],
        'grid_size_degrees': grid, 'target': metrics['target'], 'target_policy': metrics['target_policy'],
        'evaluated_at': metrics['evaluated_at'], 'split': metrics['split'],
        'saved_model_metrics': metrics['model'],
        'activity_period': {'start': str(start.date()), 'end': str(end.date())},
        'activity_scope': 'Unique satellite/time/pixel detections in cells whose 0.1-degree centres lie inside the project Thailand boundary. Known type 1/2/3 detections excluded; unknown types retained.',
        'missingness': 'Days with no regional records are unknown, not zeros. The trailing seven-day mean requires seven fully observed days.',
        'new_downloads_used': False,
        'sources_sha256': {str(p.relative_to(project)): sha256(p) for p in
            [metrics_path,dataset_path,prediction_path,coordinate_path,boundary_path]},
        'prepared_data_sha256': {p.name: sha256(p) for p in
            [data_dir/'prediction_scores.csv.gz',data_dir/'daily_activity.csv']}}
    save_json(data_dir / 'evidence.json', evidence)


def style():
    plt.rcParams.update({
        'font.family':'DejaVu Sans', 'font.size':13, 'axes.titlesize':16,
        'axes.titleweight':'semibold', 'axes.titlepad':16,
        'axes.labelsize':15, 'axes.labelpad':10,
        'xtick.labelsize':12, 'ytick.labelsize':12, 'legend.fontsize':11,
        'text.color':INK, 'axes.labelcolor':INK, 'xtick.color':INK, 'ytick.color':INK,
        'axes.edgecolor':'#AEB7BF', 'axes.linewidth':.8,
        'axes.spines.top':False, 'axes.spines.right':False,
        'axes.facecolor':'white', 'figure.facecolor':'white', 'savefig.facecolor':'white',
        'grid.color':GRID, 'grid.linewidth':.7, 'axes.axisbelow':True,
        'legend.frameon':False, 'svg.fonttype':'none', 'svg.hashsalt':'wildfire-poster-3day',
        'savefig.dpi':300, 'figure.dpi':120, 'lines.linewidth':2.3})


def title(fig, heading, subtitle):
    fig.text(.075,.955,heading,fontsize=21,weight='semibold',va='top')
    fig.text(.075,.904,subtitle,fontsize=12,color=GRAY,va='top')


def note(fig, text):
    fig.text(.075,.040,text,fontsize=10.5,color=GRAY,va='bottom',linespacing=1.5)


def probability_axes(ax):
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    ax.xaxis.set_major_locator(MultipleLocator(.2))
    ax.yaxis.set_major_locator(MultipleLocator(.2))
    ax.grid(True)


def save(fig, out, name):
    # Fixed 14 x 7 inch canvas preserves the requested 2:1 aspect ratio.
    fig.savefig(out / f'{name}.png', dpi=300)
    fig.savefig(out / f'{name}.svg', metadata={'Date':None})
    plt.close(fig)


def plot(data_dir, out):
    out.mkdir(parents=True, exist_ok=True)
    e = json.loads((data_dir/'evidence.json').read_text())
    for name, expected in e['prepared_data_sha256'].items():
        if sha256(data_dir/name) != expected:
            raise ValueError(f'Evidence checksum mismatch: {name}')
    scores = pd.read_csv(data_dir/'prediction_scores.csv.gz', parse_dates=['date'])
    daily = pd.read_csv(data_dir/'daily_activity.csv', parse_dates=['date']).set_index('date')
    y = scores.target.to_numpy(dtype=int)
    p = scores.probability.to_numpy(dtype=float)
    assert np.isin(y,[0,1]).all() and np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
    assert e['horizon_days'] == 3
    assert scores.days_until_detection.isin([-1,1,2,3]).all()
    assert np.array_equal(y==1,scores.days_until_detection.to_numpy()>0)
    assert not scores.duplicated(['date','lat_cell','lon_cell']).any()
    assert str(scores.date.min().date()) == e['split']['test_start']
    assert str(scores.date.max().date()) == e['split']['test_end']
    saved = e['saved_model_metrics']['classification']
    threshold = float(saved['threshold'])
    tn,fp,fn,tp = map(int,confusion_matrix(y,p>=threshold,labels=[0,1]).ravel())
    auc = float(roc_auc_score(y,p))
    ap = float(average_precision_score(y,p))
    prevalence = float(y.mean())
    recall, precision, false_rate = tp/(tp+fn), tp/(tp+fp), fp/(fp+tn)
    recomputed = dict(roc_auc=auc,average_precision=ap,recall=recall,precision=precision,
        false_positive_rate=false_rate,brier_score=float(brier_score_loss(y,p)),
        true_positive=tp,false_positive=fp,false_negative=fn,true_negative=tn)
    for key,value in recomputed.items():
        if not np.isclose(value,saved[key],rtol=1e-10,atol=1e-12):
            raise ValueError(f'Computed metric does not match saved evaluation: {key}')
    assert len(scores) == e['saved_model_metrics']['samples']
    start = pd.Timestamp(e['split']['test_start']).strftime('%d %b %Y').lstrip('0')
    end = pd.Timestamp(e['split']['test_end']).strftime('%d %b %Y').lstrip('0')
    subtitle = f'72-hour horizon  |  Holdout: {start} – {end}  |  n = {len(y):,} grid-day windows'
    style()

    # Figure 1: actual descriptive observations, without invented missing-day zeros.
    moving = daily[['hotspots','active_cells']].rolling(7,min_periods=7).mean()
    fig, axes = plt.subplots(2,1,figsize=(14,7),sharex=True)
    fig.subplots_adjust(left=.09,right=.975,bottom=.19,top=.80,hspace=.47)
    date_start = daily.index.min().strftime('%d %b %Y').lstrip('0')
    date_end = daily.index.max().strftime('%d %b %Y').lstrip('0')
    title(fig,'Observed daily hotspot activity',
          f'Thailand training and outcome history  |  {date_start} – {date_end}  |  0.1° grid')
    for ax,column,fill,line,panel,label in [
        (axes[0],'hotspots',BLUE,NAVY,'A. Hotspots per day','Detected hotspots'),
        (axes[1],'active_cells',LIGHT_ORANGE,ORANGE,'B. Cells with at least one hotspot','Active grid cells')]:
        ax.fill_between(daily.index,daily[column],color=fill,alpha=.65,linewidth=0,label='Daily observations')
        ax.plot(daily.index,moving[column],color=line,label='7-day moving average')
        ax.set_title(panel,loc='left',pad=10)
        ax.set_ylabel(label)
        ax.set_ylim(bottom=0)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4,integer=True))
        ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
        ax.grid(axis='y')
        ax.legend(loc='upper right',ncol=2)
    axes[1].set_xlim(daily.index.min(),daily.index.max())
    axes[1].xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    axes[1].set_xlabel('Date (UTC)')
    note(fig,'Known non-vegetation types excluded; untyped detections retained. Hotspots are not confirmed wildfires.\n'
        'Gaps denote days without regional records; the trailing 7-day mean requires seven complete days.')
    save(fig,out,'01_observed_daily_activity')

    # Figure 2: reliability and score distributions from the same held-out rows.
    bins = np.linspace(0,1,11)
    bin_id = np.searchsorted(bins[1:-1],p)
    calibration = pd.DataFrame({'bin':bin_id,'probability':p,'event':y}).groupby('bin').agg(
        count=('event','size'),mean_probability=('probability','mean'),event_frequency=('event','mean')).reset_index()
    observed,predicted = calibration_curve(y,p,n_bins=10,strategy='uniform')
    np.testing.assert_allclose(calibration.mean_probability,predicted,rtol=1e-12)
    np.testing.assert_allclose(calibration.event_frequency,observed,rtol=1e-12)
    calibration.to_csv(out/'calibration_bins.csv',index=False)
    fig,axes = plt.subplots(1,2,figsize=(14,7))
    fig.subplots_adjust(left=.075,right=.975,bottom=.19,top=.79,wspace=.27)
    title(fig,'Probability calibration and score distribution',subtitle)
    ax = axes[0]
    ax.plot([0,1],[0,1],linestyle='--',color=GRAY,linewidth=1.5,label='Perfect calibration')
    ax.plot(predicted,observed,color=NAVY,marker='o',markersize=6.5,label='Model calibration')
    ax.set_title('A. Reliability diagram',loc='left')
    ax.set(xlabel='Mean predicted probability',ylabel='Observed event frequency')
    probability_axes(ax)
    ax.legend(loc='upper left')
    ax = axes[1]
    hist_bins = np.linspace(0,1,41)
    ax.hist(p[y==0],bins=hist_bins,color=BLUE,alpha=.65,edgecolor=NAVY,linewidth=.3,
            label=f'No hotspot recorded (n = {(y==0).sum():,})')
    ax.hist(p[y==1],bins=hist_bins,color=LIGHT_ORANGE,alpha=.7,edgecolor=ORANGE,linewidth=.3,
            label=f'Hotspot recorded (n = {(y==1).sum():,})')
    neg_median,pos_median = float(np.median(p[y==0])),float(np.median(p[y==1]))
    ax.axvline(threshold,color=NAVY,linestyle='--',linewidth=1.8,label=f'Alert threshold = {threshold:.2f}')
    ax.axvline(neg_median,color='#46799E',linestyle=(0,(2,2)),linewidth=1.5,label=f'Negative median = {neg_median:.3f}')
    ax.axvline(pos_median,color=CORAL,linestyle=(0,(2,2)),linewidth=1.5,label=f'Positive median = {pos_median:.3f}')
    ax.set_title('B. Prediction score distribution',loc='left')
    ax.set(xlim=(0,1),xlabel='Predicted probability',ylabel='Frequency (log scale)',yscale='log')
    ax.set_ylim(bottom=.8,top=1e6)
    ax.xaxis.set_major_locator(MultipleLocator(.2))
    ax.grid(axis='y',which='major')
    ax.legend(loc='upper right',fontsize=10.5)
    note(fig,'Reliability: 10 equal-width bins; each marker represents all held-out windows in its bin.\n'
        'Histogram: 40 equal-width bins. Positive means a satellite hotspot recorded on days 1–3; windows overlap.')
    save(fig,out,'02_calibration_and_scores')

    # Figure 3: discrimination, with the exact event-prevalence PR baseline.
    fpr,tpr,_ = roc_curve(y,p)
    pr_precision,pr_recall,_ = precision_recall_curve(y,p)
    fig,axes = plt.subplots(1,2,figsize=(14,7))
    fig.subplots_adjust(left=.075,right=.975,bottom=.19,top=.79,wspace=.27)
    title(fig,'Model discrimination',subtitle)
    ax=axes[0]
    ax.plot(fpr,tpr,color=NAVY,label=f'Model ROC-AUC = {auc:.3f}')
    ax.plot([0,1],[0,1],color=GRAY,linestyle='--',linewidth=1.5,label='Random classification')
    ax.scatter([false_rate],[recall],s=52,color=CORAL,zorder=5,label=f'Threshold = {threshold:.2f}')
    ax.set_title('A. Receiver operating characteristic',loc='left')
    ax.set(xlabel='False positive rate',ylabel='True positive rate (recall)')
    probability_axes(ax)
    ax.legend(loc='lower right')
    ax=axes[1]
    ax.step(pr_recall,pr_precision,where='post',color=ORANGE,label=f'Model (average precision = {ap:.3f})')
    ax.axhline(prevalence,color=GRAY,linestyle='--',linewidth=1.5,label=f'Event prevalence = {prevalence:.2%}')
    ax.scatter([recall],[precision],s=52,color=NAVY,zorder=5,label=f'Threshold = {threshold:.2f}')
    ax.set_title('B. Precision–recall curve',loc='left')
    ax.set(xlabel='Recall',ylabel='Precision')
    probability_axes(ax)
    ax.legend(loc='upper right',fontsize=10.5)
    note(fig,f'At threshold {threshold:.2f}: recall = {recall:.1%}; precision = {precision:.1%}. '
        f'Positive windows: {y.sum():,} / {len(y):,}.\n'
        'Chronological holdout with a 3-day label purge. Satellite hotspot prediction; not confirmed wildfire prediction.')
    save(fig,out,'03_model_discrimination')

    # Companion table and audit receipt derive from these same arrays.
    summary=pd.DataFrame([
        ('ROC-AUC',auc,'Risk ranking; not classification accuracy'),
        ('Recall',recall,'TP / (TP + FN)'),('Precision',precision,'TP / (TP + FP)'),
        ('False positive rate',false_rate,'FP / (FP + TN)'),
        ('Average precision',ap,'Scikit-learn average_precision_score; not trapezoidal PR area'),
        ('Event prevalence',prevalence,'Positive grid-day windows / all grid-day windows'),
        ('Threshold',threshold,'Alert if probability >= threshold')],columns=['metric','value','definition'])
    summary.to_csv(out/'performance_summary.csv',index=False)
    details={'model_version':e['model_version'],'horizon_days':3,'samples':len(y),
        'split':e['split'],'metrics_recomputed':recomputed,'prevalence':prevalence,
        'threshold':threshold,'positive_median':pos_median,'negative_median':neg_median,
        'calibration_bins':10,'histogram_bins':40,
        'activity_missing_dates':daily.index[daily.hotspots.isna()].strftime('%Y-%m-%d').tolist(),
        'activity_total_eligible_detections':int(daily.hotspots.sum()),
        'activity_scope':e['activity_scope'],'data_are_actual':True,'mockups_used':False,
        'poster_reference_available':False,'new_downloads_used':False,
        'canvas_inches':[14,7],'png_dpi':300,'png_dimensions':[4200,2100],
        'software':{'python':platform.python_version(),'matplotlib':matplotlib.__version__,
            'pandas':pd.__version__,'numpy':np.__version__,'scikit_learn':sklearn.__version__}}
    save_json(out/'validation.json',details)
    print(json.dumps(details,indent=2))


def main():
    here=Path(__file__).resolve().parent
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root',type=Path,help='Rebuild portable evidence from the original project.')
    parser.add_argument('--data-dir',type=Path,default=here/'plot_data')
    parser.add_argument('--output-dir',type=Path,default=here)
    args=parser.parse_args()
    if args.project_root:
        prepare(args.project_root.resolve(),args.data_dir)
    plot(args.data_dir,args.output_dir)


if __name__=='__main__':
    main()
