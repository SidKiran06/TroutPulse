"""WatershedPulse AI — compact, four-file Pennsylvania watershed prototype.

Run: streamlit run app.py
Files beside this script: requirements.txt, daily_data.csv, sensor_data.csv.
Measured temperatures are from USGS user-downloaded exports. Live weather is
requested from Open-Meteo only when the user presses a button. No invented data.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from html import escape
from io import BytesIO
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen
import json

import folium
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

ROOT = Path(__file__).resolve().parent
TZ = 'America/New_York'
STATIONS = {
    'USGS-01549700': ('Pine Creek below Little Pine Creek', 41.27361657, -77.32411857),
    'USGS-01544500': ('Kettle Creek at Cross Fork', 41.475838, -77.825801),
    'USGS-01548030': ('Fishing Creek near Lamar', 40.980472, -77.482778),
    'USGS-01447720': ('Tobyhanna Creek near Blakeslee', 41.084808, -75.605467),
    'USGS-01447500': ('Lehigh River at Stoddartsville', 41.130363, -75.625467),
    'USGS-01451380': ('Little Lehigh Creek near Trexlertown', 40.531208, -75.600465),
    'USGS-01451800': ('Jordan Creek near Schnecksville', 40.661762, -75.626854),
    'USGS-03081000': ('Youghiogheny River below Confluence', 39.827577, -79.372535),
}
COLORS = {'check': '#DC6657', 'stale': '#9E8AC5', 'routine': '#2C9C91',
          'plan': '#D69F41', 'missing': '#8293A0'}


def load_data():
    """Read only the two flat, bundled data files; validate basic provenance."""
    daily = pd.read_csv(ROOT / 'daily_data.csv', dtype={'site_id': str}, parse_dates=['date'])
    sensors = pd.read_csv(ROOT / 'sensor_data.csv', dtype={'site_id': str})
    needed = {'site_id', 'date', 'water_temp_max_C', 'water_temp_min_C',
              'water_temp_mean_C', 'approval_status', 'source'}
    if not needed.issubset(daily.columns):
        raise ValueError(f'Daily data missing columns: {needed - set(daily.columns)}')
    if daily.duplicated(['site_id', 'date']).any():
        raise ValueError('Duplicate site/date measurements')
    if set(daily.site_id) != set(STATIONS):
        raise ValueError('Expected measured temperatures for exactly eight USGS sites')
    if not daily.water_temp_max_C.between(-5, 45).all():
        raise ValueError('Unusable daily stream temperatures')
    if not (daily.water_temp_min_C <= daily.water_temp_mean_C + .02).all():
        raise ValueError('Daily min and mean are not ordered')
    if not (daily.water_temp_mean_C <= daily.water_temp_max_C + .02).all():
        raise ValueError('Daily mean and max are not ordered')
    sensors['timestamp_local'] = pd.to_datetime(sensors['timestamp_local'], utc=True, format='mixed')
    return daily.sort_values(['site_id', 'date']), sensors.sort_values('timestamp_local')


def common_summer_comparison(daily: pd.DataFrame, threshold: float = 20.0):
    """Only June–August years with >=65 measured days per station are compared."""
    summer = daily.loc[daily.date.dt.month.isin([6, 7, 8])].copy()
    summer['year'] = summer.date.dt.year
    coverage = summer.groupby(['site_id', 'year']).size().reset_index(name='days')
    good = coverage[coverage.days >= 65]
    all_sites = set(STATIONS)
    years = sorted(y for y, group in good.groupby('year') if set(group.site_id) == all_sites)
    rows = []
    for site_id, (name, _, _) in STATIONS.items():
        part = summer[(summer.site_id == site_id) & (summer.year.isin(years))]
        n = len(part)
        warm = int((part.water_temp_max_C >= threshold).sum())
        rows.append({'site_id': site_id, 'Stream': name,
                     'Days measured': n, 'Warm days': warm if n else None,
                     '% warm days': round(100 * warm / n, 1) if n else np.nan})
    return pd.DataFrame(rows).sort_values('% warm days', ascending=False), years


def station_priority(part: pd.DataFrame, threshold: float, today: date, forecast: pd.DataFrame | None = None):
    """Evidence-aware priorities; recent vs seasonal vs air-weather planning are distinct."""
    d = part.loc[part.date.dt.date <= today].sort_values('date')
    if d.empty:
        return {'now': 'No measured data', 'now_type': 'missing',
                'week': 'Weather forecast unavailable', 'week_type': 'missing',
                'long': 'Collect measurements', 'long_type': 'missing', 'age': None}
    last = d.iloc[-1]
    age = (today - last.date.date()).days
    recent = d[d.date.dt.date >= today - timedelta(days=6)]
    recent_warm = int((recent.water_temp_max_C >= threshold).sum())
    if age > 3:
        now_msg, now_color = f'Refresh readings first (last reading {age} days ago)', 'stale'
    elif recent_warm >= 2 or (age <= 1 and last.water_temp_max_C >= threshold):
        now_msg, now_color = f'Check water temperature at this gauge soon ({recent_warm} recent warm days)', 'check'
    else:
        now_msg, now_color = 'Routine monitoring; no current heat flag at this gauge', 'routine'
    summer = d[d.date.dt.month.isin([6, 7, 8])].copy()
    summer['year'] = summer.date.dt.year
    valid_years = [y for y, g in summer.groupby('year') if len(g) >= 65]
    summer = summer[summer.year.isin(valid_years)]
    summer_pct = (100 * float((summer.water_temp_max_C >= threshold).mean())) if len(summer) else None
    if summer_pct is None:
        long_msg, long_color = 'Collect more summer measurements before ranking', 'missing'
    elif summer_pct >= 25:
        long_msg, long_color = f'Plan summer monitoring and shade/refuge field checks ({summer_pct:.0f}% flagged)', 'plan'
    else:
        long_msg, long_color = f'Continue summer monitoring ({summer_pct:.0f}% flagged)', 'routine'
    week_msg, week_color = '7-day air-weather forecast not fetched', 'missing'
    if forecast is not None and not forecast.empty:
        w = forecast.copy()
        w['date'] = pd.to_datetime(w['date']).dt.date
        w = w[(w.date > today) & (w.date <= today + timedelta(days=7))]
        hot_days = int((w.air_temp_max_C >= 28).sum())
        if w.empty:
            week_msg = 'No verified forecast days available'
        elif hot_days >= 2:
            week_msg, week_color = (f'Schedule extra water-temperature checks: {hot_days} hot AIR-temperature forecast days', 'plan')
        else:
            week_msg, week_color = ('Continue planned monitoring; no hot-air planning trigger', 'routine')
    return {'now': now_msg, 'now_type': now_color, 'week': week_msg, 'week_type': week_color,
            'long': long_msg, 'long_type': long_color, 'age': age,
            'latest': float(last.water_temp_max_C), 'latest_date': last.date.date(),
            'summer_pct': summer_pct, 'recent_warm': recent_warm}


def build_map(priorities: dict, window: str, daily: pd.DataFrame, threshold: float):
    """Water-focused map with clearer popups and a richer tooltip."""
    m = folium.Map(location=[40.8, -77.1], zoom_start=7, tiles=None, control_scale=True)
    # Water / terrain friendly basemaps
    folium.TileLayer('OpenTopoMap', name='Terrain & water', control=True).add_to(m)
    folium.TileLayer('CartoDB positron', name='Light map', control=True).add_to(m)
    folium.TileLayer('OpenStreetMap', name='Street map', control=True).add_to(m)
    key = {'Now': ('now', 'now_type'), 'Next 7 Days': ('week', 'week_type'),
           'Long Term': ('long', 'long_type')}[window]
    for sid, (name, lat, lon) in STATIONS.items():
        row = priorities[sid]
        station_daily = daily[daily.site_id == sid].sort_values('date')
        latest = station_daily.iloc[-1]
        recent_days = int(((station_daily.date >= station_daily.date.max() - pd.Timedelta(days=6)) &
                           (station_daily.water_temp_max_C >= threshold)).sum())
        summer = station_daily[station_daily.date.dt.month.isin([6, 7, 8])]
        summer_pct = None
        if len(summer):
            summer_pct = round(100 * float((summer.water_temp_max_C >= threshold).mean()), 1)
        color = COLORS[row[key[1]]]
        text = escape(row[key[0]])
        summary_lines = [
            f'<b>{escape(name)}</b>',
            f'USGS: {escape(sid)}',
            f'Latest daily max: {latest.water_temp_max_C:.1f}°C on {latest.date.date()}',
            f'Recent flagged days (7d lookback): {recent_days}',
            f'Summer warm-day rate: {summer_pct:.1f}%' if summer_pct is not None else 'Summer warm-day rate: unavailable',
            f'<b>{escape(window)} priority:</b> {text}'
        ]
        popup = ('<div style="font:14px Arial;width:280px;color:#173042;line-height:1.45">' +
                 '<br>'.join(summary_lines) +
                 f'<hr><a target="_blank" rel="noopener noreferrer" href="https://waterdata.usgs.gov/monitoring-location/{sid}/">Open official USGS station page</a></div>')
        tooltip = f'{name} | latest max {latest.water_temp_max_C:.1f}°C | {row[key[0]]}'
        folium.CircleMarker([lat, lon], radius=10, color='white', weight=2,
                            fill=True, fill_color=color, fill_opacity=.95,
                            tooltip=tooltip,
                            popup=folium.Popup(popup, max_width=340)).add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    m.fit_bounds([[39.65, -79.6], [41.7, -75.4]])
    return m


def api_json(url: str, timeout=16):
    req = Request(url, headers={'User-Agent': 'WatershedPulseAI-Educational-Prototype/1.0',
                                'Accept': 'application/json'})
    with urlopen(req, timeout=timeout) as fp:
        return json.load(fp)


def fetch_forecast(sid: str):
    _, lat, lon = STATIONS[sid]
    args = {'latitude': lat, 'longitude': lon, 'daily': 'temperature_2m_max,precipitation_sum',
            'timezone': TZ, 'forecast_days': 8}
    url = 'https://api.open-meteo.com/v1/forecast?' + urlencode(args)
    raw = api_json(url)
    items = raw.get('daily') or {}
    result = pd.DataFrame({'date': items.get('time', []),
                           'air_temp_max_C': items.get('temperature_2m_max', []),
                           'rain_mm': items.get('precipitation_sum', [])})
    if result.empty or result.air_temp_max_C.isna().all():
        raise ValueError('Open-Meteo provided no usable forecast')
    return result, url


def fetch_historical_weather(sid: str, start: date, end: date):
    _, lat, lon = STATIONS[sid]
    args = {'latitude': lat, 'longitude': lon, 'start_date': str(start), 'end_date': str(end),
            'daily': 'temperature_2m_max,precipitation_sum,shortwave_radiation_sum', 'timezone': TZ}
    url = 'https://archive-api.open-meteo.com/v1/archive?' + urlencode(args)
    raw = api_json(url, timeout=25)
    w = raw.get('daily') or {}
    table = pd.DataFrame({'date': w.get('time', []),
                          'air_temp_max_C': w.get('temperature_2m_max', []),
                          'rain_mm': w.get('precipitation_sum', []),
                          'solar_mj_m2': w.get('shortwave_radiation_sum', [])})
    if table.empty:
        raise ValueError('No historic weather returned')
    table['date'] = pd.to_datetime(table.date)
    return table, url


def evaluate_weather_model(water: pd.DataFrame, weather: pd.DataFrame):
    """Time-split retrospective model; all features observed BEFORE target day."""
    t = water[['date', 'water_temp_max_C']].copy().sort_values('date').drop_duplicates('date')
    t['yesterday_water'] = t.water_temp_max_C.shift(1)
    t['prev_three_water'] = t.water_temp_max_C.shift(1).rolling(3).mean()
    # Ensure rolling observations are consecutive days (do not bridge measurement gaps).
    t['three_day_span_ok'] = (t.date - t.date.shift(3)).dt.days.eq(3)
    w = weather[['date', 'air_temp_max_C', 'rain_mm', 'solar_mj_m2']].copy()
    w['date'] = w['date'] + pd.Timedelta(days=1)  # Yesterday's weather targets today's temp.
    merged = t.merge(w, on='date', how='inner')
    features = ['yesterday_water', 'prev_three_water', 'air_temp_max_C', 'rain_mm', 'solar_mj_m2']
    merged = merged.loc[merged.three_day_span_ok].dropna(subset=features + ['water_temp_max_C'])
    if len(merged) < 120:
        raise ValueError(f'Only {len(merged)} complete training dates; need at least 120')
    cut = int(len(merged) * .8)
    train, test = merged.iloc[:cut], merged.iloc[cut:]
    reg = RandomForestRegressor(n_estimators=120, min_samples_leaf=3, random_state=42, n_jobs=-1)
    reg.fit(train[features], train.water_temp_max_C)
    pred = reg.predict(test[features])
    baseline = test.yesterday_water.to_numpy()
    scored = test[['date', 'water_temp_max_C']].copy()
    scored['ai_prediction_C'] = pred
    scored['baseline_C'] = baseline
    return {'rows': len(merged), 'train': len(train), 'test': len(test),
            'model_mae': float(mean_absolute_error(scored.water_temp_max_C, pred)),
            'baseline_mae': float(mean_absolute_error(scored.water_temp_max_C, baseline)),
            'results': scored}


def recommended_actions(pr: dict, notes: dict):
    tasks = []
    if pr['now_type'] == 'check':
        tasks.append(('Soon: verify temperature', 'Measure at safe accessible points upstream, downstream, and near tributary junctions. Confirm sensor time and fish habitat with experts.'))
    elif pr['now_type'] == 'stale':
        tasks.append(('First: refresh measurements', 'The latest record is old; check new USGS data before scheduling an urgent visit.'))
    elif pr['long_type'] == 'plan':
        tasks.append(('Before next summer: plan monitoring', 'Identify accessible locations for temperature loggers and summer field checks.'))
    else:
        tasks.append(('Continue routine monitoring', 'Keep a consistent log and check conditions during summer warm periods.'))
    if notes.get('shade') == 'Few streamside trees':
        tasks.append(('Assess a riparian buffer', 'Photograph streamside shade and ask a landowner or conservation district whether native tree planting is suitable. Do not select a planting site from temperature alone.'))
    if notes.get('cool') == 'Possible cooler tributary':
        tasks.append(('Measure a potential cold-water refuge', 'Take paired measurements in the tributary and main stream at the same time on multiple warm days.'))
    if notes.get('erosion') == 'Active erosion':
        tasks.append(('Refer streambank erosion', 'Document from a safe location and ask a conservation district for a professional assessment. Do not construct in the stream without appropriate approval.'))
    if notes.get('clarity') == 'Unusually cloudy':
        tasks.append(('Check water quality', 'Compare with recent rainfall and ask trained volunteers to test turbidity and related indicators.'))
    if notes.get('clarity') == 'Possible pollution or fish kill':
        tasks.append(('Report the concern promptly', 'Record observations from a safe place and contact the Pennsylvania DEP or relevant authorities. Do not diagnose contamination with this app.'))
    if notes.get('litter'):
        tasks.append(('Organize an approved cleanup', 'Coordinate with local watershed groups, landowners, and experts before collecting litter or removing invasive plants.'))
    return tasks


def main():
    import streamlit as st
    import streamlit.components.v1 as components

    st.set_page_config(page_title='WatershedPulse AI | Pennsylvania', page_icon='🌊', layout='wide', initial_sidebar_state='expanded')
    st.markdown('''<style>
      :root {--pa-navy:#132D50;--pa-deep:#0B203B;--pa-gold:#E2BA6D;--pa-gold-dark:#A17625;--pa-water:#187A86;--pa-ink:#183450;--pa-pale:#F5F7FB}
      .stApp {background:var(--pa-pale)}
      /* Keep ordinary prose readable even when a visitor's Streamlit theme is dark. */
      div[data-testid="stMain"]{color:#193852}
      div[data-testid="stMain"] [data-testid="stMarkdownContainer"]>p,
      div[data-testid="stMain"] [data-testid="stMarkdownContainer"]>ul,
      div[data-testid="stMain"] [data-testid="stMarkdownContainer"]>ol{color:#26465e!important}
      div[data-testid="stMain"] h1,div[data-testid="stMain"] h2,div[data-testid="stMain"] h3{color:#193852}
      div[data-testid="stMain"] [data-testid="stCaptionContainer"]{color:#526c80!important}
      div[data-testid="stMain"] a{color:#166777}
      .block-container{max-width:1250px;padding-top:1.4rem;padding-bottom:2.2rem}
      /* South Fayette-style left navigation, in a PA flag-inspired palette. */
      section[data-testid="stSidebar"]{background:linear-gradient(180deg,#102846 0%,#193D64 100%)!important;border-right:3px solid var(--pa-gold)}
      section[data-testid="stSidebar"] *:not(svg):not(path){color:#F4F8FF}
      section[data-testid="stSidebar"] div[role="radiogroup"]>label{padding:9px 11px;margin:2px 0;border-radius:10px;border:1px solid transparent;cursor:pointer;transition:background .15s}
      section[data-testid="stSidebar"] div[role="radiogroup"]>label:hover{background:rgba(255,255,255,.10)}
      section[data-testid="stSidebar"] div[role="radiogroup"]>label:has(input:checked){background:#355273;border:1px solid #D8B56D;box-shadow:inset 4px 0 0 #E2BA6D}
      section[data-testid="stSidebar"] hr{border-color:rgba(255,255,255,.24)}
      section[data-testid="stSidebar"] .side-brand .mini,
      section[data-testid="stSidebar"] .sidebar-heading{color:#EACA91!important}
      div[data-testid="stMain"] button[data-testid="stBaseButton-primary"]{background:#173B5D!important;color:white!important;border:1px solid #E2BA6D!important}
      div[data-testid="stMain"] button[data-testid="stBaseButton-primary"]:hover{background:#245378!important}
      .side-brand{background:rgba(255,255,255,.06);padding:17px 13px;border:1px solid rgba(226,186,109,.42);border-radius:14px;margin:1px 0 16px}
      .side-brand .name{font-weight:900;letter-spacing:.015em;font-size:1.4rem;color:white}
      .side-brand .mini{font-weight:700;font-size:.74rem;color:#EACA91;letter-spacing:.1em;text-transform:uppercase;margin-bottom:5px}
      .side-brand .sub{font-size:.82rem;color:#E4ECF7;line-height:1.45;margin-top:6px}
      .sidebar-heading{color:#E5BE79;font-size:.72rem;font-weight:800;letter-spacing:.1em;margin:14px 0 6px}
      .hero{border-radius:18px;padding:26px 30px;color:#FFF;margin-bottom:15px;position:relative;overflow:hidden;background:linear-gradient(110deg,#122C4F,#1B4F66);border-bottom:5px solid #DAB267}
      .hero:after{content:'◆';position:absolute;right:25px;top:-38px;font-size:8.5rem;color:rgba(229,189,107,.17);pointer-events:none}
      .hero h1{margin:0;color:#FFF!important;font-size:2.35rem;line-height:1.2;letter-spacing:-.02em}
      .hero p{color:#DFEBF5!important;margin:.65rem 0 0;max-width:720px}
      .hero .eyebrow{color:#F6D396!important;font-size:.77rem;font-weight:800;text-transform:uppercase;letter-spacing:.12em;margin-bottom:9px}
      .page-header{margin:8px 0 20px;padding:5px 0 15px;border-bottom:2px solid #E5C27D}
      .page-header .eyebrow{color:#976B21;font-size:.77rem;font-weight:900;letter-spacing:.13em}
      .page-header h1{font-size:2rem;color:#183453!important;margin:5px 0!important}
      .page-header p{color:#50687B!important;font-size:.95rem;margin:0}
      .summary-card-row{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:13px;margin:10px 0 23px}
      .summary-card{background:#142F4E;border:1px solid #325273;border-top:4px solid #E2BA6D;border-radius:12px;padding:15px 16px;min-height:122px}
      .summary-label{color:#E6EFF9!important;font-size:.86rem;line-height:1.35;font-weight:700}
      .summary-value{color:white!important;font-size:1.52rem;font-weight:850;line-height:1.2;margin:8px 0}
      .summary-sub{color:#D4E4F0!important;font-size:.77rem;line-height:1.3}
      div[data-testid="stMetric"]{background:#142F4E;border-radius:12px;padding:14px;border-top:4px solid #E2BA6D}
      div[data-testid="stMetric"] label,div[data-testid="stMetric"] div,div[data-testid="stMetric"] p{color:white!important}
      .tp-section-heading{font-size:1.03rem;font-weight:850;color:#163E64;letter-spacing:.01em;margin:.45rem 0 .7rem}
      .tp-feature-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:13px;margin:12px 0 23px}
      .tp-feature{background:#FFF;border:1px solid #DAE3EC;border-top:4px solid #DAB267;border-radius:13px;padding:17px;min-height:145px;box-shadow:0 3px 10px rgba(27,48,72,.05)}
      .tp-feature-title{color:#183653!important;font-size:1.07rem;font-weight:850;margin:0 0 8px}
      .tp-feature-copy{color:#4E6478!important;font-size:.90rem;line-height:1.48;margin:0}
      .tp-tag{display:inline-block;font-size:.7rem;font-weight:850;letter-spacing:.06em;text-transform:uppercase;color:#937022!important;margin-bottom:8px}
      .tp-step-row{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px;margin:12px 0 20px}
      .tp-step{background:#EAF1F8;border:1px solid #D2DFEB;color:#173C59;border-radius:12px;padding:13px;line-height:1.4}
      .tp-step b{display:block;color:#133D64;margin-bottom:5px;font-size:.94rem}
      .tp-step span{display:block;color:#355D74;font-size:.85rem}
      .tp-evidence{border-left:5px solid #C69B46;background:#FFF8E9;border-radius:9px;padding:16px 19px;margin:13px 0 20px;color:#334B5A}
      .tp-evidence strong{color:#5E441F}
      .resource-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px;margin:16px 0 23px}
      .resource-card{background:#fff;border:1px solid #d7e0eb;border-left:5px solid #DAB267;border-radius:12px;padding:18px 20px;box-shadow:0 4px 14px rgba(20,47,79,.045)}
      .resource-card .org{font-size:.72rem;letter-spacing:.08em;color:#8C661F!important;text-transform:uppercase;font-weight:850;margin-bottom:5px}
      .resource-card .title{color:#183653!important;font-size:1.12rem;font-weight:850;margin-bottom:8px}
      .resource-card .desc{color:#4F6579!important;line-height:1.45;font-size:.90rem}
      .resource-card a{display:inline-block;color:#12677A!important;margin-top:13px;font-weight:800;text-decoration:none}
      .resource-card a:hover{text-decoration:underline}
      @media(max-width:950px){.summary-card-row{grid-template-columns:repeat(2,minmax(0,1fr))}.tp-feature-grid{grid-template-columns:repeat(2,minmax(0,1fr))}.tp-step-row{grid-template-columns:repeat(2,minmax(0,1fr))}}
      @media(max-width:620px){.resource-grid{grid-template-columns:1fr}.hero{padding:22px}.hero h1{font-size:1.75rem}}
      @media(max-width:480px){.summary-card-row,.tp-feature-grid,.tp-step-row{grid-template-columns:1fr}}
    </style>''', unsafe_allow_html=True)
    try:
        daily, sensors = load_data()
    except Exception as e:
        st.error(f'Could not load bundled USGS data: {e}')
        st.stop()
    from zoneinfo import ZoneInfo
    today = datetime.now(ZoneInfo(TZ)).date()
    pages = [
        '🏠 Overview',
        '🗺️ Watershed Map',
        '📍 Where to Focus',
        '📈 Stream Explorer',
        '🌡️ Thermal Watch',
        '🌱 Action Planner',
        '📸 Community Observations',
        '🧠 Weather & AI',
        '📚 Pennsylvania Resources',
        '🔎 Data Sources',
    ]
    with st.sidebar:
        st.markdown('''<div class="side-brand"><div class="mini">Pennsylvania · Freshwater Ecosystems</div>
          <div class="name">🌊 WatershedPulse AI</div>
          <div class="sub">Explore stream data. Prioritize field checks. Support smarter conservation.</div></div>''', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-heading">APP NAVIGATION</div>', unsafe_allow_html=True)
        page = st.radio('Explore WatershedPulse', pages, label_visibility='collapsed', key='watershed_page')
        st.divider()
        st.markdown('<div class="sidebar-heading">MONITORING SETTINGS</div>', unsafe_allow_html=True)
        threshold = st.slider('Warm-water screening threshold', 15.0, 26.0, 20.0, .5, format='%.1f°C')
        selected = st.selectbox('USGS monitoring station', list(STATIONS),
                                format_func=lambda key: STATIONS[key][0])
        st.caption('The threshold is an adjustable research screen, not a universal limit for freshwater species.')
        st.divider()
        st.caption('8 measured USGS sites · Pennsylvania fieldwork prototype')
        st.markdown('[USGS Water Data](https://waterdata.usgs.gov/explore/)  ·  [PA Fish & Boat](https://www.pa.gov/agencies/fishandboat/conservation)')

    station_data = {sid: daily[daily.site_id == sid].copy() for sid in STATIONS}
    forecasts = st.session_state.get('forecasts', {})
    priorities = {sid: station_priority(data, threshold, today, forecasts.get(sid))
                  for sid, data in station_data.items()}
    compared, common_years = common_summer_comparison(daily, threshold)
    last = station_data[selected].iloc[-1]
    comparison_text = ', '.join(map(str, common_years)) or 'Unavailable'
    if page == '🏠 Overview':
        st.markdown('''<div class="hero"><div class="eyebrow">Pennsylvania · Citizen science & conservation</div>
          <h1>WatershedPulse AI</h1><p>Turning measured stream temperatures into clearer monitoring priorities
          and practical next steps for Pennsylvania's freshwater ecosystems.</p></div>''', unsafe_allow_html=True)
    else:
        page_desc = {
            '🗺️ Watershed Map': 'Eight real USGS monitoring sites, priority markers, and station-level evidence.',
            '📍 Where to Focus': 'Which monitored sites deserve attention now, next week, or next summer?',
            '📈 Stream Explorer': 'Study daily water temperatures and available high-frequency readings.',
            '🌡️ Thermal Watch': 'Compare water-temperature screening results across measured summer dates.',
            '🌱 Action Planner': 'Translate monitored conditions and field notes into realistic volunteer follow-ups.',
            '📸 Community Observations': 'Log local field notes for export and later review.',
            '🧠 Weather & AI': 'Evaluate historical temperature estimates against a simple comparison method.',
            '📚 Pennsylvania Resources': 'Verified Pennsylvania agencies, programs, and fieldwork guides.',
            '🔎 Data Sources': 'See measurement provenance, coverage, and what the data represent.',
        }
        clean_title = page.split(' ', 1)[1]
        st.markdown(f'''<div class="page-header"><div class="eyebrow">WATERSHEDPULSE AI · PENNSYLVANIA</div>
          <h1>{escape(clean_title)}</h1><p>{escape(page_desc.get(page, ''))}</p></div>''',
          unsafe_allow_html=True)
    if page in ('🏠 Overview', '🗺️ Watershed Map', '📍 Where to Focus'):
        st.markdown(f'''
        <div class="summary-card-row">
          <div class="summary-card"><div class="summary-label">Measured stations</div><div class="summary-value">{len(STATIONS)}</div><div class="summary-sub">All eight have verified USGS daily temperatures</div></div>
          <div class="summary-card"><div class="summary-label">Daily measurements</div><div class="summary-value">{len(daily):,}</div><div class="summary-sub">Daily min, mean, and max water temperatures</div></div>
          <div class="summary-card"><div class="summary-label">Most recent downloaded day</div><div class="summary-value">{daily.date.max().date()}</div><div class="summary-sub">Saved USGS daily export snapshot</div></div>
          <div class="summary-card"><div class="summary-label">Fair summer comparison years</div><div class="summary-value">{comparison_text}</div><div class="summary-sub">Only overlapping summers with enough data</div></div>
        </div>''', unsafe_allow_html=True)

    if page == '🏠 Overview':
        st.subheader('WatershedPulse AI — prototype highlights')
        st.write('**Our goal:** Help watershed volunteers know **which monitored stream deserves a closer look, when to visit, and what to check**—using real temperature data rather than guesswork.')
        st.markdown('''<div class="tp-feature-grid">
          <div class="tp-feature"><div class="tp-tag">01 · Explore</div><div class="tp-feature-title">🗺️ Interactive stream map</div><p class="tp-feature-copy">Find eight monitored Pennsylvania streams. Tap a station for temperature evidence and a plain-language monitoring recommendation.</p></div>
          <div class="tp-feature"><div class="tp-tag">02 · Understand</div><div class="tp-feature-title">🌡️ Thermal Watch</div><p class="tp-feature-copy">Compare summer days reaching the selected water-temperature threshold. Find patterns worth checking, not diagnoses of fish health.</p></div>
          <div class="tp-feature"><div class="tp-tag">03 · Prioritize</div><div class="tp-feature-title">📍 Where to Focus</div><p class="tp-feature-copy">Separate recent concerns, next-week weather-based monitoring plans, and longer-term summer investigations.</p></div>
          <div class="tp-feature"><div class="tp-tag">04 · Plan</div><div class="tp-feature-title">🌱 Conservation Action Planner</div><p class="tp-feature-copy">Turn data and field observations into suggestions such as checking shade, cooler tributaries, streambank erosion, or possible riparian buffers.</p></div>
          <div class="tp-feature"><div class="tp-tag">05 · Record</div><div class="tp-feature-title">📸 Community Observations</div><p class="tp-feature-copy">Log fish sightings, water clarity, erosion, and field notes. Export the log as a CSV; entries are not shared or permanently stored yet.</p></div>
          <div class="tp-feature"><div class="tp-tag">06 · Test</div><div class="tp-feature-title">🧠 Weather & AI</div><p class="tp-feature-copy">Combine water and weather data to test next-day estimates, then compare the AI model with a simpler prediction method.</p></div>
        </div>''', unsafe_allow_html=True)
        st.markdown('''<div class="tp-section-heading">HOW A VOLUNTEER COULD USE TROUTPULSE</div>
        <div class="tp-step-row">
          <div class="tp-step"><b>1. Spot a pattern</b><span>Compare measured warm-day frequencies across monitoring stations.</span></div>
          <div class="tp-step"><b>2. Choose a field check</b><span>Prioritize a summer visit where repeated warming deserves investigation.</span></div>
          <div class="tp-step"><b>3. Collect evidence</b><span>Measure nearby stream sections, check shade and potential cool inflows.</span></div>
          <div class="tp-step"><b>4. Plan with experts</b><span>Discuss monitoring or restoration options with local watershed partners.</span></div>
        </div>''', unsafe_allow_html=True)
        st.caption('Temperature patterns flag locations to investigate. The app does not decide that a planting or restoration project is appropriate without field measurements, site conditions, and expert review.')
        if common_years and not compared.empty:
            highest = compared.iloc[0]
            lowest = compared.iloc[-1]
            yr = ', '.join(str(y) for y in common_years)
            warm_limit_f = threshold * 9 / 5 + 32
            lcol, rcol = st.columns(2)
            with lcol:
                st.metric('Most frequent warm-day readings',
                          highest['Stream'].split(' near ')[0].split(' below ')[0],
                          f"{int(highest['Warm days'])} of {int(highest['Days measured'])} summer days", delta_color='off')
            with rcol:
                st.metric('Fewest warm-day readings',
                          lowest['Stream'].split(' near ')[0],
                          f"{int(lowest['Warm days'])} of {int(lowest['Days measured'])} summer days", delta_color='off')
            st.caption(f'Days with a measured daily maximum at or above {threshold:g}°C ({warm_limit_f:.0f}°F), during June–August of {yr}. Temperature screening does not establish which species were present or whether habitat was harmed.')
            graph = compared.sort_values('% warm days', ascending=True)
            b = go.Figure(go.Bar(x=graph['% warm days'], y=graph['Stream'], orientation='h',
                                 text=[f"{r['% warm days']:.1f}% ({int(r['Warm days'])}/{int(r['Days measured'])})" for _, r in graph.iterrows()],
                                 textposition='outside', marker_color='#128b96'))
            b.update_layout(height=370, margin=dict(l=8,r=75,t=16,b=10),
                            xaxis=dict(title=f'% of measured summer days ≥ {threshold:g}°C', range=[0,112]),
                            yaxis_title=None)
            st.plotly_chart(b, use_container_width=True)
            st.markdown('**From measurements to conservation priorities**')
            hottest_name = escape(str(highest['Stream']))
            coolest_name = escape(str(lowest['Stream']))
            st.markdown(f'''<div class="tp-evidence"><strong>Suggested summer fieldwork priority: {hottest_name}.</strong>
            Its monitoring station recorded frequent warm summer daily maxima. Volunteers could check upstream and downstream temperatures, compare possible cooler tributary inflows, and document streamside shade during safe, planned visits.</div>''', unsafe_allow_html=True)
            st.write(f"**Comparison site: {lowest['Stream']}.** The gauge showed fewer flagged summer days, making it useful for comparing temperature patterns—not proof that its habitat is healthy.")
            st.write('**If shade is limited or erosion is visible:** Record those conditions in Community Observations and consult landowners, watershed groups, and conservation specialists before considering planting or streambank projects.')
        else:
            st.info('A fair summer comparison requires enough shared summer dates across all eight stations.')
        st.markdown('**Quick TV demonstration (three steps)**')
        st.write('**1. Show the finding:** Point to the warm-day comparison above, then open Map and tap a station.')
        st.write('**2. Show the decision:** Open Where to Focus and Action Planner. Select few streamside trees or a possible cool tributary to show how suggested field checks change.')
        st.write('**3. Show the testing:** Open Weather & AI and compare model error against the simple baseline. Explain that the forecast capability is still being improved.')
        st.caption('Currently working: eight-station measured-data comparisons, map, thermal screening, fieldwork suggestions, and session-only volunteer observations. Planned or still being validated: reliable next-day water-temperature forecasts, statewide site-specific planting recommendations, and permanent shared observation storage.')

    if page == '🗺️ Watershed Map':
        st.subheader('Watershed monitoring map')
        period = st.radio('View priorities for', ['Now', 'Next 7 Days', 'Long Term'], horizontal=True, key='mapperiod')
        st.caption('This map is for stream-monitoring decisions. Click a station to see the latest temperature, warm-day history, and what volunteers should investigate next.')
        mp = build_map(priorities, period, daily, threshold)
        components.html(mp.get_root().render(), height=560, scrolling=False)
        rows = []
        for sid, (name, _, _) in STATIONS.items():
            series = station_data[sid].sort_values('date')
            latest = series.iloc[-1]
            recent_flagged = int(((series.date >= series.date.max() - pd.Timedelta(days=6)) & (series.water_temp_max_C >= threshold)).sum())
            summer = series[series.date.dt.month.isin([6, 7, 8])]
            summer_pct = round(100 * float((summer.water_temp_max_C >= threshold).mean()), 1) if len(summer) else np.nan
            rows.append({
                'Stream': name,
                'Latest daily max (°C)': round(float(latest.water_temp_max_C), 1),
                'Latest measurement': str(latest.date.date()),
                'Recent flagged days (7d)': recent_flagged,
                'Summer warm-day %': summer_pct,
                f'{period} message': priorities[sid][{'Now':'now','Next 7 Days':'week','Long Term':'long'}[period]]
            })
        st.markdown('**Map summary table**')
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        st.caption('Default basemap is OpenTopoMap so water and terrain are easier to read. Marker colors show monitoring priority, not proof of stream damage or planting locations.')

    if page == '📍 Where to Focus':
        st.subheader('Where should volunteers focus?')
        st.write('Temperature readings suggest **where to verify conditions**; field assessments determine which conservation projects are appropriate.')
        view = st.selectbox('Choose a planning time frame', ['Now', 'Next 7 Days', 'Longer Term'], key='focus_period')
        msg, color = {
            'Now': ('now', 'now_type'),
            'Next 7 Days': ('week', 'week_type'),
            'Longer Term': ('long', 'long_type'),
        }[view]
        if msg == 'week':
            st.info('These are **air-weather** planning signals, not predicted water temperatures.')
            if st.button('Fetch 7-day weather for all eight streams', type='primary'):
                failures = []
                new = dict(forecasts)
                with st.spinner('Checking Open-Meteo forecasts...'):
                    with ThreadPoolExecutor(max_workers=4) as pool:
                        jobs = {pool.submit(fetch_forecast, sid): sid for sid in STATIONS}
                        for job in as_completed(jobs):
                            sid = jobs[job]
                            try:
                                new[sid] = job.result()[0]
                            except Exception as exc:
                                failures.append(f'{STATIONS[sid][0]}: {exc}')
                st.session_state['forecasts'] = new
                st.session_state['forecast_errors'] = failures
                st.rerun()
            for error in st.session_state.get('forecast_errors', []):
                st.warning(error[:200])
        order = {'check': 0, 'plan': 1, 'stale': 2, 'routine': 3, 'missing': 4}
        records = sorted(STATIONS, key=lambda sid: (order[priorities[sid][color]],
                     -(priorities[sid].get('summer_pct') or 0)))
        for sid in records:
            with st.container(border=True):
                st.markdown(f'**{STATIONS[sid][0]}**')
                st.write(priorities[sid][msg])
                if msg == 'now':
                    st.caption(f"Last measured daily maximum: {priorities[sid]['latest']:.1f}°C on {priorities[sid]['latest_date']}. Not necessarily today's temperature.")
        if msg == 'long':
            st.caption('Warmer stations deserve closer *investigation*, not automatic tree planting. Managed rivers and different habitat types require additional evidence.')

    if page == '📈 Stream Explorer':
        st.subheader('Stream Explorer')
        chosen = station_data[selected]
        st.write(f'**{STATIONS[selected][0]}** · {len(chosen):,} measured days · {chosen.date.min().date()} to {chosen.date.max().date()}')
        span = st.select_slider('Show the past', options=[30, 90, 365, 730, 1825, 6000], value=365,
                                format_func=lambda x: f'{x} days')
        part = chosen[chosen.date >= chosen.date.max() - pd.Timedelta(days=span)]
        fig = go.Figure()
        for col, label in [('water_temp_max_C', 'Daily maximum'),
                           ('water_temp_mean_C', 'Daily mean'), ('water_temp_min_C', 'Daily minimum')]:
            fig.add_trace(go.Scatter(x=part.date, y=part[col], name=label, mode='lines'))
        fig.add_hline(y=threshold, line_dash='dash', annotation_text=f'{threshold:g}°C screen')
        fig.update_layout(height=385, yaxis_title='Water temperature (°C)', margin=dict(l=5,r=5,t=12,b=8))
        st.plotly_chart(fig, use_container_width=True)
        recent = sensors[sensors.site_id == selected]
        if not recent.empty:
            st.markdown('**Downloaded high-frequency sensor measurements**')
            sfig = go.Figure(go.Scatter(x=recent.timestamp_local, y=recent.water_temp_C, mode='lines', name='15-minute readings'))
            sfig.update_layout(height=245, yaxis_title='Water temperature (°C)', margin=dict(l=5,r=5,t=8,b=8))
            st.plotly_chart(sfig, use_container_width=True)
            st.caption(f"Last downloaded sensor timestamp: {recent.timestamp_local.max().tz_convert(TZ)}. These are saved readings, not a continuous live feed.")
        else:
            st.caption('No recent 15-minute sensor export bundled for this station; the daily readings above are verified.')

    if page == '🌡️ Thermal Watch':
        st.subheader('Thermal Watch')
        st.write(f'Days with a **daily maximum at or above {threshold:g}°C** are flagged for closer review. A flagged day does not prove fish were present or harmed.')
        sample = station_data[selected].copy()
        sample['Year'] = sample.date.dt.year
        summary = sample.groupby('Year').agg(Days_measured=('date','size'),
            Flagged_days=('water_temp_max_C', lambda x: int((x >= threshold).sum()))).reset_index()
        summary['% flagged'] = (100 * summary.Flagged_days / summary.Days_measured).round(1)
        summary['Coverage'] = summary.Days_measured.apply(lambda n: 'Partial year' if n < 350 else 'Near-full year')
        st.dataframe(summary.rename(columns={'Days_measured':'Days measured','Flagged_days':'Flagged days'}),
                     hide_index=True, use_container_width=True)
        st.markdown('**Compare the eight stations fairly using the same summers**')
        st.caption('Included years: ' + (', '.join(map(str, common_years)) if common_years else 'none with sufficient overlap') + '. Summer is June–August; dates missing from a station are excluded from its denominator.')
        st.dataframe(compared[['Stream','Warm days','Days measured','% warm days']], hide_index=True,
                     use_container_width=True)
        bar = go.Figure(go.Bar(x=compared['Stream'], y=compared['% warm days']))
        bar.update_layout(height=350, xaxis_tickangle=-25, yaxis_title='% of measured summer days reaching threshold')
        st.plotly_chart(bar, use_container_width=True)

    if page == '🌱 Action Planner':
        st.subheader('Conservation Action Planner')
        st.write('This combines temperature screening with **volunteer field observations** to suggest what to investigate, not to authorize restoration work.')
        c1, c2 = st.columns(2)
        with c1:
            shade = st.selectbox('Streamside shade', ['Not checked','Few streamside trees','Plenty of shade'])
            erosion = st.selectbox('Streambank erosion', ['Not checked','No clear erosion','Active erosion'])
            cool = st.selectbox('Tributary or spring', ['Not checked','Possible cooler tributary','None identified'])
        with c2:
            clarity = st.selectbox('Water clarity', ['Not checked','Looks clear','Unusually cloudy','Possible pollution or fish kill'])
            litter = st.checkbox('Litter or invasive plants observed')
            st.caption('Input notes are unverified until field partners review them.')
        notes = {'shade': shade, 'erosion': erosion, 'cool': cool, 'clarity': clarity, 'litter': litter}
        advice = recommended_actions(priorities[selected], notes)
        st.markdown(f"**Station screening result:** {priorities[selected]['now']}")
        for title, explanation in advice:
            with st.container(border=True):
                st.markdown('**' + title + '**')
                st.write(explanation)
        st.caption('Tree planting, habitat construction, or stream alterations require site suitability, landowner permission, and sometimes professional design or permits.')

    if page == '📸 Community Observations':
        st.subheader('Community Observations')
        st.write('Record field notes, fish sightings, stream clarity, and erosion. This prototype keeps reports **only for the current browser session**; export CSV to save them.')
        if 'observations' not in st.session_state:
            st.session_state['observations'] = []
        with st.form('observation'):
            col1, col2 = st.columns(2)
            with col1:
                obs_site = st.selectbox('Stream', list(STATIONS), format_func=lambda sid: STATIONS[sid][0])
                obs_day = st.date_input('Observation date', value=today, max_value=today)
                kind = st.selectbox('Observation', ['Fish sighting','Water clarity','Streambank erosion','Riparian vegetation','Possible cooler tributary','Litter / invasive plants','Other'])
            with col2:
                general_location = st.text_input('General location (no private addresses)')
                description = st.text_area('What did you observe?', max_chars=1500)
                photo = st.file_uploader('Optional photo (session-only)', type=['jpg','jpeg','png'])
            submitted = st.form_submit_button('Add observation')
            if submitted:
                if not description.strip():
                    st.error('Please add an observation description.')
                else:
                    st.session_state.observations.append({'date': str(obs_day), 'stream': STATIONS[obs_site][0],
                        'type': kind, 'general_location': general_location.strip(),
                        'description': description.strip(), 'photo_filename': photo.name if photo else '',
                        'verified': 'No — volunteer report'})
                    st.success('Observation added to this session. Export CSV to keep a copy.')
        if st.session_state.observations:
            obs = pd.DataFrame(st.session_state.observations)
            st.dataframe(obs, hide_index=True, use_container_width=True)
            st.download_button('Export observation log (CSV)', obs.to_csv(index=False).encode('utf-8'),
                               'watershedpulse_observations.csv', 'text/csv')
        else:
            st.info('No observations entered in this session yet.')

    if page == '🧠 Weather & AI':
        st.subheader('Weather integration and AI validation')
        st.write('The app can retrieve **historical weather** to test whether a model improves on a simple water-temperature baseline. It does not claim a validated live stream-temperature forecast.')
        if st.button('Fetch historical weather and evaluate model for selected stream', type='primary'):
            try:
                with st.spinner('Downloading historical gridded weather from Open-Meteo and training...'):
                    series = station_data[selected]
                    # Use recent available years so one API call stays reasonably small.
                    start = max(series.date.min().date(), series.date.max().date() - timedelta(days=1095))
                    end = series.date.max().date()
                    hist, url = fetch_historical_weather(selected, start, end)
                    score = evaluate_weather_model(series, hist)
                    st.session_state['model_score'] = (selected, score, url)
            except Exception as exc:
                st.error(f'Could not validate model: {exc}. This may be an API/network issue; the USGS dashboard remains usable.')
        cached = st.session_state.get('model_score')
        if cached and cached[0] == selected:
            score = cached[1]
            p1, p2, p3 = st.columns(3)
            p1.metric('AI mean absolute error', f"{score['model_mae']:.2f} °C")
            p2.metric('Simple baseline error', f"{score['baseline_mae']:.2f} °C")
            p3.metric('Held-out dates', f"{score['test']:,}")
            if score['model_mae'] < score['baseline_mae']:
                st.success('On held-out historical dates, the model beat the simple baseline.')
            else:
                st.info('On held-out dates, the simple baseline was as good or better. We will not claim the AI improves accuracy.')
            scored = score['results']
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=scored.date, y=scored.water_temp_max_C, name='Actual water temperature'))
            fig.add_trace(go.Scatter(x=scored.date, y=scored.ai_prediction_C, name='AI retrospective prediction'))
            fig.update_layout(height=350, yaxis_title='Daily max (°C)')
            st.plotly_chart(fig, use_container_width=True)
            st.caption('Train on earlier ~80% and evaluate on later ~20%, with only prior-day measurements used as predictors. Open-Meteo weather is gridded historical data, not an on-site sensor.')
        else:
            st.info('Weather-dependent model not validated in this session. Press the button when the deployed app has internet access.')
        if forecasts:
            with st.expander('Loaded next-week AIR-temperature forecasts'):
                for sid, table in forecasts.items():
                    st.markdown(f"**{STATIONS[sid][0]}**")
                    st.dataframe(table, hide_index=True, use_container_width=True)

    if page == '📚 Pennsylvania Resources':
        st.markdown('**Explore trusted Pennsylvania guidance for monitoring, habitat protection, and volunteer projects.**')
        st.write('These are external public resources. WatershedPulse helps volunteers decide what to investigate; landowners, conservation districts, and specialists decide whether a physical project is appropriate.')
        resources = [
            ('Monitoring & data', 'USGS Water Data for the Nation', 'Explore official monitoring stations and confirm dates, measured parameters, and recent readings.', 'https://waterdata.usgs.gov/explore/'),
            ('Fish & aquatic habitat', 'PA Fish and Boat Commission — Habitat Management', 'Learn about stream and lake habitats, aquatic biodiversity, and professionally planned habitat improvements.', 'https://www.pa.gov/agencies/fishandboat/conservation/habitat-management'),
            ('Riparian buffers', 'PA DCNR — Forest Buffers Along Waterways', 'Understand how native streamside vegetation filters runoff, reduces erosion, and provides shade and wildlife habitat.', 'https://www.pa.gov/agencies/dcnr/conservation/water/riparian-buffers'),
            ('Fieldwork & volunteering', 'Penn State — Master Watershed Stewards', 'Explore training and volunteer programs for stream cleanups, water-quality checks, education, and buffer planting.', 'https://extension.psu.edu/programs/watershed-stewards/about'),
            ('Habitat assistance', 'PA Fish and Boat — Technical Assistance', 'Find guidance on planning aquatic habitat and riparian corridor improvement with trained partners.', 'https://www.pa.gov/services/fishandboat/technical-assistance-program'),
            ('Environmental reporting', 'PA DEP — Report Incidents and Complaints', 'For suspected pollution or environmental incidents, use the official reporting process rather than relying on the app.', 'https://www.pa.gov/agencies/dep/report-incidents-and-complaints'),
            ('Species & stewardship', 'PA Fish and Boat — Aquatic Habitats & Watersheds', 'Explore educational materials about Pennsylvania freshwater food webs, watersheds, and healthy stream conditions.', 'https://www.pa.gov/agencies/fishandboat/education/activities-education-portal/habitat-watersheds-pollution'),
            ('Funding & support', 'PA Fish and Boat — Cooperative Habitat Improvement', 'Learn how community groups and landowners can seek technical help with qualifying habitat restoration.', 'https://www.pa.gov/services/fishandboat/cooperative-habitat-improvement-program'),
        ]
        topic = st.selectbox('Find resources about', ['All resources', 'Monitoring & data', 'Fish & aquatic habitat', 'Riparian buffers', 'Fieldwork & volunteering', 'Habitat assistance', 'Environmental reporting', 'Species & stewardship', 'Funding & support'])
        shown = [r for r in resources if topic == 'All resources' or r[0] == topic]
        cols = st.columns(2)
        for i, (category, title, description, url) in enumerate(shown):
            with cols[i % 2]:
                st.markdown(f'''<div class="resource-card"><div class="org">{escape(category)}</div>
                  <div class="title">{escape(title)}</div><div class="desc">{escape(description)}</div>
                  <a href="{escape(url)}" target="_blank" rel="noopener noreferrer">Explore official resource ↗</a></div>''', unsafe_allow_html=True)
        st.markdown('**A responsible fieldwork workflow**')
        st.markdown('1. **Check measurements:** Use USGS data to identify temperature patterns.\n'
                    '2. **Verify on site:** Measure shade, tributary temperatures, water clarity, and bank conditions safely.\n'
                    '3. **Ask local partners:** Discuss findings with a watershed group, conservation district, or PFBC/DCNR specialist.\n'
                    '4. **Plan appropriate work:** Consider native buffers, monitoring, or cleanups only after confirming site needs and permissions.')
        st.info('Current app scope: temperature records at eight stream/river monitoring stations. This does not measure whole watersheds or individual lakes.')

    if page == '🔎 Data Sources':
        st.subheader('Data sources and limitations')
        st.write('**USGS:** Water-temperature parameter 00010, daily maximum 00001, daily minimum 00002, daily mean 00003, supplied as official USGS downloads. Recent sensor readings are bundled for three stations only.')
        st.write('**Weather:** Open-Meteo historical archive and air-temperature forecast APIs are accessed when requested; weather data are not bundled in these four files.')
        st.write('**Important:** A monitoring gauge describes one location, not a whole river. A 20°C threshold is an exploratory screen, not a fish-health diagnosis or proof of habitat damage. Managed rivers need separate interpretation.')
        manifest = []
        for sid, data in station_data.items():
            manifest.append({'Station': STATIONS[sid][0], 'USGS ID': sid,
                             'First measurement': str(data.date.min().date()),
                             'Last measurement': str(data.date.max().date()),
                             'Daily dates': len(data),
                             'Source': f'https://waterdata.usgs.gov/monitoring-location/{sid}/'})
        st.dataframe(pd.DataFrame(manifest), hide_index=True, use_container_width=True)
        st.download_button('Export verified daily measurements', daily.to_csv(index=False).encode('utf-8'),
                           file_name='watershedpulse_daily_export.csv', mime='text/csv')
        st.write('**Relevant conservation background:** [Penn State Extension riparian buffers](https://extension.psu.edu/what-are-riparian-buffers) · [PA DCNR riparian buffers](https://www.pa.gov/agencies/dcnr/conservation/water/riparian-buffers) · [PA Trout Unlimited](https://patrout.org/)')
    st.divider()
    st.caption('WatershedPulse AI • Pennsylvania freshwater ecosystem prototype • Fieldwork screening, not official restoration advice')


if __name__ == '__main__':
    main()
