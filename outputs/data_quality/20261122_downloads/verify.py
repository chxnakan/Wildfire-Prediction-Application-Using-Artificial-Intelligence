"""Read-only FIRMS download audit; stream inputs, retain aggregate evidence only.

Run from repository root with PYTHONPATH=. .venv/bin/python <this file>.
SQLite scratch space is removed on completion; source CSVs are never installed.
"""
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile
import hashlib
import json
import sqlite3

import numpy as np
import pandas as pd
from src.geography import polygons, in_thailand
from src.pipeline import SATELLITES, observation_paths

ROOT = Path.cwd()
OUT = Path(__file__).resolve().parent
INPUTS = [Path('/home/xsmax/Downloads') / name for name in (
    'DL_FIRE_J1V-C2_852088.zip', 'DL_FIRE_J2V-C2_852089.zip',
    'DL_FIRE_M-C61_852087.zip', 'DL_FIRE_SV-C2_852090.zip')]
KEYS = ['satellite', 'observed_at', 'latitude', 'longitude']


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def country_mask(lat, lon):
    result = np.zeros(len(lat), dtype=bool)
    for (w, s, e, n), rings in polygons():
        idx = np.flatnonzero((lon >= w) & (lon <= e) & (lat >= s) & (lat <= n))
        if not len(idx):
            continue
        x, y = lon[idx], lat[idx]
        selected = np.zeros(len(idx), dtype=bool)
        for number, ring in enumerate(rings):
            inside = np.zeros(len(idx), dtype=bool)
            xj, yj = ring[-1]
            for xi, yi in ring:
                if yi != yj:
                    inside ^= ((yi > y) != (yj > y)) & (x < (xj - xi) * (y - yi) / (yj - yi) + xi)
                xj, yj = xi, yi
            selected = inside if number == 0 else selected & ~inside
        result[idx] |= selected
    return result


def normalize(raw):
    times = raw.acq_time.astype(str).str.replace(r'\.0$', '', regex=True).str.zfill(4)
    ts = pd.to_datetime(raw.acq_date + ' ' + times, format='%Y-%m-%d %H%M', errors='coerce')
    sat = raw.satellite if 'satellite' in raw else raw.dataset
    return pd.DataFrame({'satellite': sat.replace(SATELLITES),
        'observed_at': ts.dt.strftime('%Y-%m-%d %H:%M'),
        'latitude': pd.to_numeric(raw.latitude, errors='coerce'),
        'longitude': pd.to_numeric(raw.longitude, errors='coerce')})


def main():
    result = {'date_of_check': '2026-11-22', 'archives': [], 'files': [],
              'method': 'Exact observation keys: normalized satellite + UTC minute + latitude + longitude. Thailand points and grid centres use the bundled province polygons.',
              'limitations': ['Zero detection rows do not establish a satellite observation outage or absence of fire.',
                  'Archive CRC verifies integrity, not authenticity against a NASA server-side checksum.',
                  'Different satellites can detect the same fire; they are separate observations, not separate fire incidents.']}
    daily = Counter()
    samples_checked = 0
    with TemporaryDirectory(prefix='audit-', dir=OUT) as scratch:
        db = sqlite3.connect(str(Path(scratch) / 'keys.sqlite'))
        db.execute('PRAGMA journal_mode=OFF')
        db.execute('PRAGMA synchronous=OFF')
        db.execute('PRAGMA cache_size=-12000')
        db.execute('PRAGMA temp_store=FILE')
        schema = '(sat TEXT, ts TEXT, lat REAL, lon REAL, flags INTEGER, types INTEGER, PRIMARY KEY(sat,ts,lat,lon)) WITHOUT ROWID'
        db.execute('CREATE TABLE old ' + schema)
        db.execute('CREATE TABLE new ' + schema)
        old_raw = 0
        for path in observation_paths(ROOT):
            for raw in pd.read_csv(path, dtype={'acq_time': str}, chunksize=50000):
                old_raw += len(raw)
                norm = normalize(raw).dropna()
                for date, count in norm.observed_at.str[:10].value_counts().items():
                    daily[('existing', date)] += int(count)
                db.executemany('INSERT OR IGNORE INTO old VALUES (?,?,?,?,1,0)', norm.itertuples(index=False, name=None))
            db.commit()
        result['existing'] = {'raw_rows': old_raw, 'unique_detections': db.execute('SELECT COUNT(*) FROM old').fetchone()[0],
            'date_range': db.execute('SELECT MIN(ts),MAX(ts) FROM old').fetchone()}
        print('Existing data indexed', flush=True)
        file_index = 0
        for path in INPUTS:
            with ZipFile(path) as z:
                bad = z.testzip()
                if bad:
                    raise ValueError(f'Corrupt ZIP member: {path.name}/{bad}')
                result['archives'].append({'path': str(path), 'sha256': digest(path), 'crc': 'pass',
                    'uncompressed_bytes': sum(i.file_size for i in z.infolist())})
                for name in z.namelist():
                    if not name.endswith('.csv'):
                        continue
                    flag = 1 << file_index
                    file_index += 1
                    p = {'archive': path.name, 'member': name, 'rows': 0, 'nulls': Counter(),
                        'invalid': Counter(), 'satellites': Counter(), 'instruments': Counter(),
                        'types': Counter(), 'confidence': Counter(), 'versions': Counter(),
                        'outside_thailand_points': 0, 'outside_thailand_grid_centres': 0,
                        'frp_min': float('inf'), 'frp_max': float('-inf'), 'date_min': '9999', 'date_max': '',
                        'lat_min': 90., 'lat_max': -90., 'lon_min': 180., 'lon_max': -180.}
                    with z.open(name) as f:
                        for raw in pd.read_csv(f, dtype={'acq_time': str}, chunksize=50000):
                            p['columns'] = list(raw.columns)
                            p['rows'] += len(raw)
                            p['nulls'].update({c: int(v) for c, v in raw.isna().sum().items() if v})
                            norm = normalize(raw)
                            invalid = norm.isna().any(axis=1) | ~norm.latitude.between(-90,90) | ~norm.longitude.between(-180,180)
                            p['invalid']['key_or_coordinates'] += int(invalid.sum())
                            p['invalid']['future_timestamp'] += int((norm.observed_at > '2026-11-22 23:59').sum())
                            conf = raw.confidence.astype(str).str.lower()
                            if 'M-C61' in name:
                                valid_conf = pd.to_numeric(conf, errors='coerce').between(0,100)
                            else:
                                valid_conf = conf.isin(['l','n','h'])
                            p['invalid']['confidence'] += int((~valid_conf).sum())
                            p['invalid']['daynight'] += int((~raw.daynight.isin(['D','N'])).sum())
                            for col in ['frp','brightness','bright_t31','scan','track']:
                                val = pd.to_numeric(raw[col], errors='coerce')
                                bad_value = val.isna() | ~np.isfinite(val) | (val < 0 if col == 'frp' else val <= 0)
                                p['invalid'][col] += int(bad_value.sum())
                            types = pd.to_numeric(raw.get('type', pd.Series(np.nan,index=raw.index)),errors='coerce')
                            p['invalid']['type'] += int((types.notna() & ~types.isin([0,1,2,3])).sum())
                            p['types'].update(types.fillna(-1).astype(int).astype(str))
                            p['satellites'].update(raw.satellite.astype(str))
                            p['instruments'].update(raw.instrument.astype(str))
                            p['confidence'].update(conf)
                            p['versions'].update(raw.version.astype(str))
                            p['date_min'] = min(p['date_min'], str(raw.acq_date.min()))
                            p['date_max'] = max(p['date_max'], str(raw.acq_date.max()))
                            for col, short in [('latitude','lat'),('longitude','lon'),('frp','frp')]:
                                p[short+'_min'] = min(p[short+'_min'],float(raw[col].min()))
                                p[short+'_max'] = max(p[short+'_max'],float(raw[col].max()))
                            coords = norm.loc[~invalid]
                            inside = country_mask(coords.latitude.to_numpy(), coords.longitude.to_numpy())
                            p['outside_thailand_points'] += int((~inside).sum())
                            # Validate the vectorized polygon check against the project's scalar implementation.
                            for i in np.linspace(0,len(coords)-1,min(20,len(coords)),dtype=int):
                                row = coords.iloc[i]
                                assert inside[i] == in_thailand(row.latitude,row.longitude)
                                samples_checked += 1
                            cells = np.rint(coords[['latitude','longitude']].to_numpy() / .1).astype(int)
                            unique, inverse = np.unique(cells,axis=0,return_inverse=True)
                            grid_inside = np.array([in_thailand(a*.1,b*.1) for a,b in unique])[inverse]
                            p['outside_thailand_grid_centres'] += int((~grid_inside).sum())
                            for date, count in raw.acq_date.value_counts().items():
                                daily[(name,date)] += int(count)
                            for date,count in raw.loc[coords.index[grid_inside],'acq_date'].value_counts().items():
                                daily[('new_thai_grid',date)] += int(count)
                                if 'M-C61' not in name:
                                    daily[('new_viirs_thai_grid',date)] += int(count)
                            records = norm.loc[~invalid].copy()
                            records['flags'] = flag
                            records['types'] = types.loc[~invalid].map({0:1,1:2,2:4,3:8}).fillna(0).astype(int)
                            db.executemany('INSERT INTO new VALUES (?,?,?,?,?,?) ON CONFLICT(sat,ts,lat,lon) DO UPDATE SET flags=flags|excluded.flags, types=types|excluded.types',records.itertuples(index=False,name=None))
                    db.commit()
                    unique_count = db.execute('SELECT COUNT(*) FROM new WHERE flags & ? != 0',(flag,)).fetchone()[0]
                    p['duplicate_observation_keys'] = p['rows']-p['invalid']['key_or_coordinates']-unique_count
                    result['files'].append(p)
                    print(name, p['rows'], p['date_min'], p['date_max'], 'invalid', dict(p['invalid']), flush=True)
        result['new_unique_detections'] = db.execute('SELECT COUNT(*) FROM new').fetchone()[0]
        result['new_raw_rows'] = sum(p['rows'] for p in result['files'])
        result['new_duplicate_observation_keys'] = result['new_raw_rows']-result['new_unique_detections']
        result['conflicting_known_type_keys'] = db.execute('SELECT COUNT(*) FROM new WHERE types NOT IN (0,1,2,4,8)').fetchone()[0]
        result['overlap_with_existing'] = db.execute('SELECT COUNT(*) FROM new INNER JOIN old USING(sat,ts,lat,lon)').fetchone()[0]
        result['new_to_project'] = result['new_unique_detections']-result['overlap_with_existing']
        result['polygon_crosscheck_points'] = samples_checked
        for label in ['existing','new_thai_grid','new_viirs_thai_grid']:
            dates = sorted(date for (group,date), n in daily.items() if group == label and n)
            calendar = pd.date_range(dates[0], dates[-1]).strftime('%Y-%m-%d').tolist()
            missing = sorted(set(calendar)-set(dates))
            result[label+'_coverage'] = {'first':dates[0],'last':dates[-1], 'days_with_records':len(dates),
                'days_without_records':len(missing), 'missing_dates':missing}
        old_missing = result['existing_coverage']['missing_dates']
        result['old_no_record_days_now_with_viirs_records'] = sum(daily[('new_viirs_thai_grid',date)]>0 for date in old_missing)
        result['recent_viirs_no_record_dates_since_2025'] = [d for d in result['new_viirs_thai_grid_coverage']['missing_dates'] if d>='2025-01-01']
        result['new_detections_by_year'] = dict(db.execute('SELECT substr(ts,1,4),COUNT(*) FROM new GROUP BY substr(ts,1,4)').fetchall())
        result['new_detections_by_satellite'] = dict(db.execute('SELECT sat,COUNT(*) FROM new GROUP BY sat').fetchall())
        db.close()
    (OUT/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    pd.DataFrame([{'source':s,'date':d,'rows':n} for (s,d),n in sorted(daily.items())]).to_csv(OUT/'daily_counts.csv',index=False)
    print('SAVED', OUT/'results.json', flush=True)


if __name__ == '__main__':
    main()
