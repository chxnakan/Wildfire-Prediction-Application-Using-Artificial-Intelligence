"""Follow up the duplicate finding and summarize coverage without importing data."""
from pathlib import Path
from zipfile import ZipFile
import json
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
r = json.loads((OUT/'results.json').read_text())
path = Path('/home/xsmax/Downloads/DL_FIRE_SV-C2_852090.zip')
member = 'fire_archive_SV-C2_852090.csv'
keys = ['satellite','acq_date','acq_time','latitude','longitude']
hashes = []
with ZipFile(path) as z, z.open(member) as f:
    for raw in pd.read_csv(f,dtype={'acq_time':str},chunksize=50000):
        hashes.append(pd.util.hash_pandas_object(raw[keys],index=False).to_numpy())
values, counts = np.unique(np.concatenate(hashes),return_counts=True)
repeated = values[counts>1]
del hashes, values, counts
rows = []
with ZipFile(path) as z, z.open(member) as f:
    for raw in pd.read_csv(f,dtype={'acq_time':str},chunksize=50000):
        h = pd.util.hash_pandas_object(raw[keys],index=False).to_numpy()
        rows.append(raw[np.isin(h,repeated)])
duplicates = pd.concat(rows,ignore_index=True)
# Hashes locate candidates only; verify actual field equality before reporting.
assert int(duplicates.duplicated(keys).sum()) == 2
duplicate_details = {'member':member, 'repeated_key_rows':len(duplicates),
    'extra_rows_by_observation_key':int(duplicates.duplicated(keys).sum()),
    'extra_rows_exact_all_columns':int(duplicates.duplicated().sum()),
    'rows':json.loads(duplicates.to_json(orient='records'))}
assert duplicate_details['extra_rows_by_observation_key']==r['new_duplicate_observation_keys']
daily=pd.read_csv(OUT/'daily_counts.csv',dtype={'date':str})
viirs=daily[daily.source.eq('new_viirs_thai_grid')].set_index('date').rows
start,end='2025-01-31','2026-04-22'
calendar=pd.date_range(start,end).strftime('%Y-%m-%d')
old_missing=r['existing_coverage']['missing_dates']
tail=pd.date_range('2026-11-01','2026-11-16').strftime('%Y-%m-%d')
summary={
    'duplicate_detail':duplicate_details,
    'outside_project_country_polygon_rows':sum(f['outside_thailand_points'] for f in r['files']),
    'outside_project_thailand_grid_rows':sum(f['outside_thailand_grid_centres'] for f in r['files']),
    'known_nonvegetation_rows':sum(sum(f['types'].get(str(t),0) for t in [1,2,3]) for f in r['files']),
    'untyped_nrt_rows':sum(f['types'].get('-1',0) for f in r['files']),
    'old_missing_days_still_without_new_viirs': [d for d in old_missing if viirs.get(d,0)==0],
    'latest_viirs_daily_counts':{d:int(viirs.get(d,0)) for d in tail},
    'archive_summary':[{ 'archive':a['path'],
        'rows':sum(f['rows'] for f in r['files'] if f['archive']==Path(a['path']).name),
        'first':min(f['date_min'] for f in r['files'] if f['archive']==Path(a['path']).name),
        'last':max(f['date_max'] for f in r['files'] if f['archive']==Path(a['path']).name)} for a in r['archives']],
    'code_compatibility':{
        'brightness_column_supported':True,
        'satellite_names_supported':True,
        'nrt_missing_type_supported':True,
        'static_context_rejects_dates_before':'2023-01-01',
        'weather_cache_for_published_model_ends':'2026-01-21',
        'direct_import_or_retraining_performed':False},
    'reference_urls':['https://firms.modaps.eosdis.nasa.gov/download/',
        'https://firms.modaps.eosdis.nasa.gov/content/active_fire/',
        'https://open-meteo.com/en/docs/previous-runs-api']}
(OUT/'details.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
