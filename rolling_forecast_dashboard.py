"""
Rolling Forecast Variance Dashboard
Strictly Computes Matrix Variance Natively across Multiple Demand Sheets
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, date
import io
import plotly.graph_objects as go

st.set_page_config(page_title="Rolling Forecast Variance Dashboard", layout="wide")

BUCKET_THRESHOLDS = {
    "0-6 Weeks": 5.0,    
    "7-10 Weeks": 10.0,  
    "11-13 Weeks": 15.0, 
    "14+ Weeks": 20.0    
}

bg, bg2, txt, txt2, accent, green, red, amber, border, grid_c, plot_bg = (
    "#ffffff", "#f4f6f9", "#1a1f2e", "#5a6478", "#2563eb", 
    "#16a34a", "#dc2626", "#d97706", "#d1d9e6", "rgba(0,0,0,0.04)", "rgba(0,0,0,0)"
)

COMPANIES = {
    "JL":  "Jakson Limited",
    "PL":  "Powerica Limited",
    "SPL": "Sudhir Power Limited",
}

st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');
  html,body,[class*="css"]{{font-family:'Inter',sans-serif;background:{bg};color:{txt};}}
  .stApp{{background:{bg};}}
  .block-container{{padding:2rem 2.5rem;max-width:1400px;}}
  section[data-testid="stSidebar"]{{background:{bg2};border-right:1px solid {border};}}
  section[data-testid="stSidebar"] .block-container{{padding:1.5rem 1rem;}}
  header[data-testid="stHeader"]{{background:transparent!important;}}
  .kpi-grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:1.2rem 0 1.8rem;}}
  .kpi{{background:{bg2};border:1px solid {border};border-radius:10px;padding:16px 18px;}}
  .kpi-label{{font-size:10px;font-weight:600;color:{txt2};text-transform:uppercase;letter-spacing:0.08em;margin-bottom:6px;}}
  .kpi-value{{font-size:26px;font-weight:600;color:{txt};line-height:1;}}
  .kpi-value.danger{{color:{red};}} .kpi-value.success{{color:{green};}} .kpi-value.accent{{color:{accent};}}
  .section-title{{font-size:13px;font-weight:600;color:{txt};margin:1.8rem 0 0.8rem;padding-bottom:6px;border-bottom:1px solid {border};}}
  .info-bar{{background:{bg2};border:1px solid {border};border-radius:8px;padding:10px 16px;font-size:12px;color:{txt2};margin-bottom:1.2rem;line-height:1.8;}}
  .info-bar b{{color:{txt};}} .info-bar .sep{{margin:0 10px;color:{border};}}
  .bias-box{{background:{bg2};border:1px solid {border};border-radius:10px;padding:16px;margin-bottom:1rem;}}
  .bias-row{{display:flex;gap:16px;align-items:center;margin-bottom:8px;}}
  .bias-label{{font-size:11px;color:{txt2};width:95px;}}
  .bias-val{{font-size:13px;font-weight:500;}}
  .bias-ok{{color:{green};}} .bias-over{{color:{red};font-weight:600;}}
  div[data-baseweb="select"]>div{{background:{bg2}!important;border-color:{border}!important;color:{txt}!important;border-radius:8px!important;}}
  div[data-baseweb="select"] span{{color:{txt}!important;}}
  input[type="text"]{{background:{bg2}!important;color:{txt}!important;border-color:{border}!important;border-radius:8px!important;}}
  .stButton>button{{background:{accent};color:white;border:none;border-radius:8px;font-weight:500;font-size:13px;padding:8px 22px;width:100%;}}
  .stButton>button:hover{{opacity:0.9;}}
  .stDownloadButton>button{{background:{bg2};color:{txt};border:1px solid {border};border-radius:8px;font-weight:500;font-size:13px;padding:8px 18px;}}
  footer{{display:none;}} .stDeployButton{{display:none;}}
  [data-testid="stWidgetLabel"]{{color:#1a1f2e!important;font-size:12px!important;font-weight:500!important;}}
  div[data-testid="stDataFrame"]{{border-radius:10px;overflow:hidden;border:1px solid {border};}}
</style>
""", unsafe_allow_html=True)


def parse_any_date(v):
    if isinstance(v, (datetime, date)):
        return v if isinstance(v, date) else v.date()
    if isinstance(v, (int, float)):
        if v > 40000:
            try: return (datetime(1899, 12, 30) + pd.to_timedelta(int(v), unit='D')).date()
            except Exception: pass
    if isinstance(v, str):
        v_str = v.strip()
        for fmt in ["%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%d-%b-%Y", "%d-%m-%Y", "%d/%m/%Y", "%m/%d/%Y"]:
            try: return datetime.strptime(v_str, fmt).date()
            except ValueError: pass
    return None


def find_header_index(raw):
    for i in range(min(15, len(raw))):
        row_vals = [str(x).lower().strip() for x in raw.iloc[i].tolist()]
        if any(keyword in val for keyword in ['coolpac', 'item', 'material', 'part'] for val in row_vals):
            return i
    return -1


def process_raw_dataframe(raw, name, ref_date):
    header_idx = find_header_index(raw)
    if header_idx == -1 or len(raw) <= header_idx + 1:
        return None, []
    
    headers = ["" if pd.isna(x) else str(x).strip() for x in raw.iloc[header_idx].tolist()]

    item_col = next((i for i, h in enumerate(headers) if 'coolpac' in h.lower()), None)
    if item_col is None:
        item_col = next((i for i, h in enumerate(headers) if any(x in h.lower() for x in ['item', 'part', 'material'])), 4)

    model_col  = next((i for i, h in enumerate(headers) if 'engine model' in h.lower()), 3)
    rating_col = next((i for i, h in enumerate(headers) if 'rating' in h.lower()), 1)
    phase_col  = next((i for i, h in enumerate(headers) if 'phase' in h.lower()), None)
    family_col = next((i for i, h in enumerate(headers) if 'family' in h.lower()), None)

    date_col_map = {}
    for i, h in enumerate(headers):
        parsed_dt = parse_any_date(h)
        if parsed_dt:
            date_col_map[pd.Timestamp(parsed_dt).normalize()] = i

    if not date_col_map: return None, []
    
    ref_ts = pd.Timestamp(ref_date).normalize()
    valid_dates_count = sum(1 for d in date_col_map.keys() if d >= ref_ts)
    if valid_dates_count == 0: return None, []

    data = raw.iloc[header_idx + 1:].reset_index(drop=True)
    mask = data.iloc[:, item_col].notna() & data.iloc[:, item_col].astype(str).str.strip().str.startswith('A0')
    data = data[mask].reset_index(drop=True)

    family_list = []
    if family_col is not None and family_col < len(data.columns):
        family_list = data.iloc[:, family_col].astype(str).str.strip().tolist()
    else:
        models_tmp = data.iloc[:, model_col].astype(str).str.strip().tolist() if model_col < len(data.columns) else []
        for m in models_tmp:
            if '-' in m: family_list.append(m.split('-')[0].strip())
            elif ' ' in m: family_list.append(m.split(' ')[0].strip())
            else: family_list.append(m if m else "N/A")

    rows = {
        'item':   data.iloc[:, item_col].astype(str).str.strip().tolist(),
        'model':  data.iloc[:, model_col].astype(str).str.strip().tolist() if model_col < len(data.columns) else [""] * len(data),
        'rating': data.iloc[:, rating_col].astype(str).str.strip().tolist() if rating_col < len(data.columns) else [""] * len(data),
        'phase':  data.iloc[:, phase_col].astype(str).str.strip().tolist() if (phase_col is not None and phase_col < len(data.columns)) else ["N/A"] * len(data),
        'family': family_list if len(family_list) == len(data) else ["N/A"] * len(data)
    }
    
    for dt, col_idx in date_col_map.items():
        if col_idx < len(data.columns):
            rows[dt] = pd.to_numeric(data.iloc[:, col_idx], errors='coerce').fillna(0).tolist()
        else:
            rows[dt] = [0] * len(data)

    return pd.DataFrame(rows), sorted(date_col_map.keys())


def load_forecast(file_bytes, file_name, ref_date):
    if file_name.lower().endswith('.csv'):
        try: 
            raw = pd.read_csv(io.BytesIO(file_bytes), header=None, encoding='utf-8')
        except Exception: 
            raw = pd.read_csv(io.BytesIO(file_bytes), header=None, encoding='latin1')
        return process_raw_dataframe(raw, file_name, ref_date)
    else:
        xf = pd.ExcelFile(io.BytesIO(file_bytes))
        best_df, best_dates = None, []
        max_valid_dates = -1
        ref_ts = pd.Timestamp(ref_date).normalize()

        for sheet in xf.sheet_names:
            try:
                raw = pd.read_excel(io.BytesIO(file_bytes), sheet_name=sheet, header=None)
                df_parsed, dates_parsed = process_raw_dataframe(raw, file_name, ref_date)
                if df_parsed is not None:
                    valid_dates_count = sum(1 for d in dates_parsed if d >= ref_ts)
                    if valid_dates_count > max_valid_dates:
                        max_valid_dates = valid_dates_count
                        best_df = df_parsed
                        best_dates = dates_parsed
            except Exception:
                continue
        if best_df is not None: return best_df, best_dates
        raise ValueError(f"No valid forecast matrices found inside {file_name} for the selected date window.")


@st.cache_data(show_spinner=False)
def cached_forecast(fb, fn, ref_date):
    return load_forecast(fb, fn, ref_date)


def get_n_week_forecast(df, dates, ref_date, n_weeks):
    ref_ts = pd.Timestamp(ref_date).normalize()
    future = sorted([d for d in dates if d >= ref_ts])[:n_weeks]
    if not future: return pd.DataFrame(), []
    result = df[['item','model','rating','phase','family']].copy()
    for i, d in enumerate(future):
        result[f'w{i+1}'] = pd.to_numeric(df[d], errors='coerce').fillna(0).tolist() if d in df.columns else [0]*len(result)
    result['total'] = result[[f'w{i+1}' for i in range(len(future))]].sum(axis=1)
    return result, future


def base_layout(height=300, **kwargs):
    return dict(
        plot_bgcolor=plot_bg, paper_bgcolor=plot_bg,
        font=dict(family="Inter", color=txt, size=12),
        margin=dict(l=0,r=0,t=10,b=0), height=height,
        hoverlabel=dict(bgcolor=bg2, bordercolor=border, font_color=txt),
        xaxis=dict(gridcolor=grid_c, tickfont=dict(color=txt2), linecolor=border),
        yaxis=dict(gridcolor=grid_c, tickfont=dict(color=txt2), linecolor=border),
        **kwargs
    )

# ── Reusable Component Renderer ───────────────────────────────────────────────
def render_variance_view(df, nw, code, info_context=None):
    if info_context:
        st.markdown(f"""
        <div class="info-bar">
          <b>Demand old File:</b> {info_context['f1_name']}<span class="sep">|</span>
          <b>Demand new File:</b> {info_context['f2_name']}<span class="sep">|</span>
          <b>Reference Date:</b> {info_context['ref'].strftime('%d %b %Y')}<span class="sep">|</span>
          <b>Weeks Compared:</b> {nw}
        </div>
        """, unsafe_allow_html=True)

    st.markdown(f"<div class='section-title'>Schedule Net Variance by Horizon Bucket</div>", unsafe_allow_html=True)
    b1, b2, b3 = st.columns(3)
    for col, bname in zip([b1,b2,b3], ['0-6','7-10','11-13']):
        vol1 = int(df[f'b1_{bname}'].sum())
        vol2 = int(df[f'b2_{bname}'].sum())
        net_var_pct = ((vol2 - vol1) / vol1 * 100) if vol1 != 0 else (0.0 if vol2 == 0 else float('inf'))
        
        mapped_limit = BUCKET_THRESHOLDS.get(f"{bname} Weeks", 10.0)
        color_cls = "bias-over" if abs(net_var_pct) > mapped_limit else "bias-ok"
        flag = f" ⚠ OVER TARGET (>{mapped_limit}%)" if abs(net_var_pct) > mapped_limit else " ✓ WITHIN TARGET"
        
        col.markdown(f"""
        <div class="bias-box">
          <div style="font-size:12px;font-weight:600;color:{txt};margin-bottom:8px;">Horizon Window Block: {bname} Weeks</div>
          <div class="bias-row"><div class="bias-label">Demand old Vol</div><div class="bias-val">{vol1:,} Units</div></div>
          <div class="bias-row"><div class="bias-label">Demand new Vol</div><div class="bias-val">{vol2:,} Units</div></div>
          <div class="bias-row"><div class="bias-label">Net Shift</div><div class="bias-val {color_cls}">{net_var_pct:+.1f}%{flag}</div></div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown(f"<div class='section-title'>Family & Horizon Window Variance Summary Deck</div>", unsafe_allow_html=True)
    
    fam_bkt_rows = []
    for fam in sorted(df['family'].dropna().unique()):
        fam_df = df[df['family'] == fam]
        for bname in ['0-6','7-10','11-13','14+']:
            v1, v2 = fam_df[f'b1_{bname}'].sum(), fam_df[f'b2_{bname}'].sum()
            diff = v2 - v1
            pct = (diff / v1 * 100) if v1 != 0 else (0.0 if v2 == 0 else float('inf'))
            fam_bkt_rows.append({
                'Product Family Group': fam, 'Horizon Window': f"{bname} Weeks",
                'Demand old Volume': int(v1), 'Demand new Volume': int(v2),
                'Net Shift': int(diff), 'Variance %': pct
            })
    
    fam_bkt_df = pd.DataFrame(fam_bkt_rows)
    
    def style_family_matrix(row):
        val, mapped_limit = row['Variance %'], BUCKET_THRESHOLDS.get(row['Horizon Window'], 10.0)
        if val == float('inf') or abs(val) > mapped_limit:
            return [f"background-color:rgba(220,38,38,0.07); font-weight:500;"] * len(row)
        return [""] * len(row)

    def color_variance_cells(val):
        if isinstance(val, (int, float)):
            if val > 0: return f"color:{green}; font-weight:600;"
            if val < 0: return f"color:{red}; font-weight:600;"
        return ""

    if not fam_bkt_df.empty:
        styled_fam = fam_bkt_df.style.apply(style_family_matrix, axis=1)
        styled_fam = styled_fam.map(color_variance_cells, subset=['Net Shift', 'Variance %'])
        styled_fam = styled_fam.format({'Demand old Volume': '{:,.0f}', 'Demand new Volume': '{:,.0f}', 'Net Shift': '{:+,.0f}', 'Variance %': '{:+.1f}%'}, na_rep='0')
        st.dataframe(styled_fam, use_container_width=True, height=280)
    else:
        st.info("No localized material variant groups map into active buckets.")

    increased, decreased, net_chg = int((df['total_diff']>0).sum()), int((df['total_diff']<0).sum()), int(df['total_diff'].sum())
    st.markdown(f"""
    <div class="kpi-grid">
      <div class="kpi"><div class="kpi-label">Tracked Components</div><div class="kpi-value accent">{len(df):,}</div></div>
      <div class="kpi"><div class="kpi-label">Increased Targets</div><div class="kpi-value success">{increased:,}</div></div>
      <div class="kpi"><div class="kpi-label">Decreased Targets</div><div class="kpi-value danger">{decreased:,}</div></div>
      <div class="kpi"><div class="kpi-label">Gross Net Deviation</div><div class="kpi-value {'danger' if net_chg<0 else 'success' if net_chg>0 else ''}">{net_chg:+,}</div></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"<div class='section-title'>Horizon Forecast Volume Comparison Timeline</div>", unsafe_allow_html=True)
    wk_labels  = [f"Wk {w}" for w in range(1,nw+1)]
    f1_tots    = [df[f'f1_w{w}'].sum() for w in range(1,nw+1)]
    f2_tots    = [df[f'f2_w{w}'].sum() for w in range(1,nw+1)]
    diff_tots  = [df[f'd_w{w}'].sum()  for w in range(1,nw+1)]

    fig = go.Figure()
    fig.add_trace(go.Bar(name="Demand old", x=wk_labels, y=f1_tots, marker_color=accent, opacity=0.8))
    fig.add_trace(go.Bar(name="Demand new", x=wk_labels, y=f2_tots, marker_color=green,  opacity=0.8))
    fig.add_trace(go.Scatter(name="Delta Variance Track", x=wk_labels, y=diff_tots, mode="lines+markers", line=dict(color=amber,width=2.5)))
    fig.add_hline(y=0, line_color=border, line_width=1)
    fig.update_layout(**base_layout(300, barmode="group", legend=dict(orientation="h",y=1.08,x=0,bgcolor="rgba(0,0,0,0)")))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(f"<div class='section-title'>Interactive Component Matrix Analytics</div>", unsafe_allow_html=True)
    fc1,fc2,fc3,fc4 = st.columns(4)
    sel_model  = fc1.multiselect("Filter Engine Model", sorted(df['model'].dropna().unique().tolist()), placeholder="All", key=f"{code}_model")
    sel_rating = fc2.multiselect("Filter Operational Rating", sorted(df['rating'].dropna().unique().tolist()), placeholder="All", key=f"{code}_rating")
    sel_phase  = fc3.multiselect("Filter Phase Line", sorted(df['phase'].dropna().unique().tolist()), placeholder="All", key=f"{code}_phase")
    sel_status = fc4.multiselect("Filter Tracking Status", ["On plan","Increased","Decreased"], placeholder="All", key=f"{code}_status")
    search     = st.text_input("Component Match Query", placeholder="Enter specific string...", key=f"{code}_search")

    fdf = df.copy()
    if sel_model:  fdf = fdf[fdf['model'].isin(sel_model)]
    if sel_rating: fdf = fdf[fdf['rating'].isin(sel_rating)]
    if sel_phase:  fdf = fdf[fdf['phase'].isin(sel_phase)]
    if sel_status: fdf = fdf[fdf['status'].isin(sel_status)]
    if search: fdf = fdf[fdf['item'].str.lower().str.contains(search.lower(),na=False)|fdf['model'].str.lower().str.contains(search.lower(),na=False)]

    st.markdown(f"<div style='font-size:12px;color:{txt2};margin-bottom:0.5rem'>{len(fdf):,} matching items isolated</div>", unsafe_allow_html=True)

    show_wk  = st.checkbox("Toggle Week-by-Week Data", key=f"{code}_wk")
    show_bkt = st.checkbox("Toggle Segment Horizon Summary Blocks", key=f"{code}_bkt", value=True)

    fdf_s = fdf.sort_values('total_diff', key=lambda x: x.abs(), ascending=False)
    base_c = ['item','model','rating','phase','family','total_f1','total_f2','total_diff','pct_diff','status']
    base_n = ['Item ID','Engine Model','Rating Spec','Phase','Family','Demand old Vol','Demand new Vol','Net Deviation','Δ %','State']
    wk_c, wk_n, bkt_c, bkt_n = [], [], [], []
    
    if show_wk:
        wk_c = [f'f1_w{w}' for w in range(1,nw+1)] + [f'f2_w{w}' for w in range(1,nw+1)] + [f'd_w{w}' for w in range(1,nw+1)]
        wk_n = [f'Demand old Wk {w}' for w in range(1,nw+1)] + [f'Demand new Wk {w}' for w in range(1,nw+1)] + [f'Δ Wk {w}' for w in range(1,nw+1)]
    if show_bkt:
        for bname in ['0-6','7-10','11-13','14+']:
            bkt_c += [f'b1_{bname}',f'b2_{bname}',f'bd_{bname}',f'bp_{bname}']
            bkt_n += [f'Demand old {bname} Wk',f'Demand new {bname} Wk',f'Δ {bname} Wk',f'Δ% {bname} Wk']

    display = fdf_s[base_c + wk_c + bkt_c].copy()
    display.columns = base_n + wk_n + bkt_n

    def color_rows(row):
        if row['State']=='Increased': return [f"background-color:rgba(22,163,74,0.05)"]*len(row)
        if row['State']=='Decreased': return [f"background-color:rgba(220,38,38,0.05)"]*len(row)
        return [""]*len(row)

    fmt = {'Demand old Vol':'{:,.0f}','Demand new Vol':'{:,.0f}','Net Deviation':'{:+,.0f}','Δ %':'{:+.1f}%'}
    var_cols = ['Net Deviation']
    for w in range(1,nw+1):
        fmt[f'Demand old Wk {w}'] = fmt[f'Demand new Wk {w}'] = '{:,.0f}'
        fmt[f'Δ Wk {w}'] = '{:+,.0f}'
        var_cols.append(f'Δ Wk {w}')
    for bname in ['0-6','7-10','11-13','14+']:
        fmt[f'Demand old {bname} Wk'] = fmt[f'Demand new {bname} Wk'] = '{:,.0f}'
        fmt[f'Δ {bname} Wk'], fmt[f'Δ% {bname} Wk'] = '{:+,.0f}', '{:+.1f}%'
        var_cols.extend([f'Δ {bname} Wk', f'Δ% {bname} Wk'])

    var_cols_in = [c for c in var_cols if c in display.columns]
    styled = display.style.apply(color_rows, axis=1)
    if var_cols_in: styled = styled.map(color_variance_cells, subset=var_cols_in)
    st.dataframe(styled.format(fmt, na_rep='0'), use_container_width=True, height=500)

    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine='openpyxl') as writer:
        display.to_excel(writer, sheet_name='Summary_Variance', index=False)
        if not fam_bkt_df.empty: fam_bkt_df.to_excel(writer, sheet_name='Family_Bucket_Breakdown', index=False)
    st.download_button(f"Download Consolidated {code} Matrix Report (.xlsx)", data=buf.getvalue(),
        file_name=f"{code}_forecast_variance_{datetime.today().strftime('%d%b%Y')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


# ── Sidebar Control Panel ─────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"<div style='font-size:14px;font-weight:600;color:{txt};margin-bottom:16px;'>Control Centre</div>", unsafe_allow_html=True)
    st.markdown(f"<div style='font-size:11px;font-weight:600;color:{txt2};text-transform:uppercase;letter-spacing:0.05em;margin-bottom:6px;'>Horizon Scope Range</div>", unsafe_allow_html=True)
    n_weeks = st.slider("weeks", 1, 26, 13, label_visibility="collapsed")
    st.markdown(f"<div style='font-size:11px;color:{txt2};margin-bottom:16px;'>{n_weeks} weeks selected</div>", unsafe_allow_html=True)

    for code, name in COMPANIES.items():
        with st.expander(f"{name} ({code})", expanded=False):
            st.markdown(f"<div style='font-size:10px;color:{txt2};margin-bottom:4px;'>Demand old</div>", unsafe_allow_html=True)
            f1 = st.file_uploader(f"demand_old_{code}", type=["xlsx","xls","csv"], key=f"{code}_f1", label_visibility="collapsed")
            st.markdown(f"<div style='font-size:10px;color:{txt2};margin-top:6px;margin-bottom:4px;'>Demand new</div>", unsafe_allow_html=True)
            f2 = st.file_uploader(f"demand_new_{code}", type=["xlsx","xls","csv"], key=f"{code}_f2", label_visibility="collapsed")
            st.markdown(f"<div style='font-size:10px;color:{txt2};margin-top:6px;margin-bottom:4px;'>Reference Baseline Date</div>", unsafe_allow_html=True)
            ref_date = st.date_input(f"ref_{code}", value=date.today(), key=f"{code}_date", label_visibility="collapsed")

    st.write("")
    run = st.button("Run Analysis")

# ── Processing & Execution ────────────────────────────────────────────────────
st.markdown(f"""
<div style="padding-bottom:1.2rem;border-bottom:1px solid {border};margin-bottom:1.5rem;">
  <div style="font-size:20px;font-weight:600;color:{txt};">Rolling Forecast Variance Dashboard</div>
  <div style="font-size:12px;color:{txt2};">Jakson · Powerica · Sudhir Power — Native Cross-Demand Comparison Runway</div>
</div>
""", unsafe_allow_html=True)

if run:
    results = {}
    errors  = []

    for code, name in COMPANIES.items():
        f1, f2, ref = st.session_state.get(f"{code}_f1"), st.session_state.get(f"{code}_f2"), st.session_state.get(f"{code}_date", date.today())
        if not f1 and not f2: continue
        if not f1 or not f2:
            errors.append(f"{name}: Both baseline demand records are required.")
            continue

        try:
            df1, dates1 = cached_forecast(f1.read(), f1.name, ref)
            df2, dates2 = cached_forecast(f2.read(), f2.name, ref)

            fc1, wk1 = get_n_week_forecast(df1, dates1, ref, n_weeks)
            fc2, wk2 = get_n_week_forecast(df2, dates2, ref, n_weeks)

            if fc1.empty or fc2.empty:
                errors.append(f"{name}: No forecast data coordinates intersect with baseline date.")
                continue

            all_items = sorted(set(fc1['item'].tolist()) | set(fc2['item'].tolist()))
            i1, i2, n = fc1.set_index('item'), fc2.set_index('item'), min(len(wk1), len(wk2), n_weeks)

            rows = []
            for item in all_items:
                r1 = i1.loc[item] if item in i1.index else None
                r2 = i2.loc[item] if item in i2.index else None
                row = {
                    'item':item,
                    'model':r1['model'] if r1 is not None else (r2['model'] if r2 is not None else ''),
                    'rating':r1['rating'] if r1 is not None else (r2['rating'] if r2 is not None else ''),
                    'phase':r1['phase'] if r1 is not None else (r2['phase'] if r2 is not None else ''),
                    'family':r1['family'] if r1 is not None else (r2['family'] if r2 is not None else 'N/A')
                }
                for w in range(1, n+1):
                    v1 = float(r1[f'w{w}']) if r1 is not None and f'w{w}' in r1.index else 0.0
                    v2 = float(r2[f'w{w}']) if r2 is not None and f'w{w}' in r2.index else 0.0
                    row[f'f1_w{w}'], row[f'f2_w{w}'], row[f'd_w{w}'] = round(v1), round(v2), round(v2 - v1)
                    
                t1, t2 = sum(row.get(f'f1_w{w}',0) for w in range(1,n+1)), sum(row.get(f'f2_w{w}',0) for w in range(1,n+1))
                row['total_f1'], row['total_f2'], row['total_diff'] = round(t1), round(t2), round(t2 - t1)
                row['pct_diff'] = round((t2-t1)/t1*100,1) if t1!=0 else (0.0 if t2==0 else float('inf'))

                bkt = {'0-6':(1,6),'7-10':(7,10),'11-13':(11,13),'14+':(14,9999)}
                for bname,(bs,be) in bkt.items():
                    bt1, bt2 = sum(row.get(f'f1_w{w}',0) for w in range(1,n+1) if bs<=w<=be), sum(row.get(f'f2_w{w}',0) for w in range(1,n+1) if bs<=w<=be)
                    row[f'b1_{bname}'], row[f'b2_{bname}'], row[f'bd_{bname}'] = round(bt1), round(bt2), round(bt2-bt1)
                    row[f'bp_{bname}'] = round((bt2-bt1)/bt1*100,1) if bt1!=0 else (0.0 if bt2==0 else float('inf'))

                row['status'] = 'On plan' if row['total_diff']==0 else ('Increased' if row['total_diff']>0 else 'Decreased')
                rows.append(row)

            results[code] = {'df': pd.DataFrame(rows), 'name': name, 'ref': ref, 'n': n, 'f1_name': f1.name, 'f2_name': f2.name}
        except Exception as e:
            errors.append(f"{name}: {str(e)}")

    st.session_state['results'], st.session_state['n_weeks'], st.session_state['run_time'] = results, n_weeks, datetime.now().strftime("%d %b %Y %H:%M")
    for e in errors: st.error(e)

if 'results' not in st.session_state or not st.session_state['results']:
    st.stop()

results, n_weeks, run_time = st.session_state['results'], st.session_state['n_weeks'], st.session_state.get('run_time','—')
st.markdown(f"<div class='info-bar'><b>Last run:</b> {run_time}<span class='sep'>|</span><b>Horizon Runway Width:</b> {n_weeks} weeks</div>", unsafe_allow_html=True)

# ── Dynamic Tab Generation ────────────────────────────────────────────────────
tab_names = [f"{COMPANIES[c]} ({c})" for c in results]
show_consolidated = (len(results) == len(COMPANIES))

if show_consolidated:
    tab_names.append("Consolidated Network")

tabs = st.tabs(tab_names)

for i, (code, res) in enumerate(results.items()):
    with tabs[i]:
        render_variance_view(res['df'], res['n'], code, res)

if show_consolidated:
    with tabs[-1]:
        st.markdown(f"<div class='info-bar'><b>Consolidated View Active:</b> All underlying demand structures integrated.</div>", unsafe_allow_html=True)
        all_dfs = [res['df'] for res in results.values()]
        master_df = pd.concat(all_dfs, ignore_index=True)
        
        agg_funcs = {col: 'sum' for col in master_df.columns if col not in ['item', 'model', 'rating', 'phase', 'family', 'status', 'pct_diff', 'bp_0-6', 'bp_7-10', 'bp_11-13', 'bp_14+']}
        agg_df = master_df.groupby(['item', 'model', 'rating', 'phase', 'family']).agg(agg_funcs).reset_index()
        
        agg_df['pct_diff'] = np.where(agg_df['total_f1'] == 0, np.where(agg_df['total_f2'] == 0, 0.0, float('inf')), ((agg_df['total_f2'] - agg_df['total_f1']) / agg_df['total_f1']) * 100).round(1)
        for bname in ['0-6', '7-10', '11-13', '14+']:
            agg_df[f'bp_{bname}'] = np.where(agg_df[f'b1_{bname}'] == 0, np.where(agg_df[f'b2_{bname}'] == 0, 0.0, float('inf')), ((agg_df[f'b2_{bname}'] - agg_df[f'b1_{bname}']) / agg_df[f'b1_{bname}']) * 100).round(1)
        
        agg_df['status'] = np.where(agg_df['total_diff'] == 0, 'On plan', np.where(agg_df['total_diff'] > 0, 'Increased', 'Decreased'))
        
        render_variance_view(agg_df, n_weeks, "MASTER_CONSOLIDATED", None)
