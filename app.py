import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

# --- CONFIG ---
st.set_page_config(layout="wide", page_title="US Infant Mortality | BMED Study")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }
    h1, h2, h3 {
        font-family: 'DM Serif Display', serif !important;
    }
    .main-title {
        font-family: 'DM Serif Display', serif;
        font-size: 2.4rem;
        color: #1a1a2e;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        font-family: 'DM Sans', sans-serif;
        font-size: 0.95rem;
        color: #6b7280;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 300;
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }
    .metric-box {
        background: linear-gradient(135deg, #f8f9ff 0%, #eef2ff 100%);
        border-radius: 12px;
        padding: 1rem 1.5rem;
        border-left: 4px solid #6366f1;
        margin-bottom: 1rem;
    }
    .section-label {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #9ca3af;
        font-weight: 500;
        margin-bottom: 0.3rem;
    }
</style>
""", unsafe_allow_html=True)

# --- DATA LOADING ---
@st.cache_data
def load_data():
    file_name = "UNdata_Export_20260417_223711907.csv"
    if not os.path.exists(file_name):
        st.error(f"❌ File `{file_name}` not found. Make sure it's in the same directory as this script.")
        st.stop()

    df = pd.read_csv(file_name, low_memory=False)
    df.columns = [str(c).strip() for c in df.columns]

    # --- Smart column detection ---
    def find_col(keywords, exclude=None):
        exclude = exclude or []
        for col in df.columns:
            col_l = col.lower()
            if any(k.lower() in col_l for k in keywords) and not any(e.lower() in col_l for e in exclude):
                return col
        return None

    year_col    = find_col(['year'])
    sex_col     = find_col(['sex'])
    area_col    = find_col(['area', 'residence'], exclude=['country'])
    value_col   = find_col(['value', 'number'])
    country_col = find_col(['country', 'area'], exclude=['residence'])

    rename_map = {}
    if year_col:    rename_map[year_col]    = 'Year'
    if sex_col:     rename_map[sex_col]     = 'Sex'
    if area_col:    rename_map[area_col]    = 'Area'
    if value_col:   rename_map[value_col]   = 'Value'
    if country_col: rename_map[country_col] = 'Country'

    df = df.rename(columns=rename_map)

    # Keep only US rows
    if 'Country' in df.columns:
        df = df[df['Country'].astype(str).str.contains('United States', na=False)]

    df['Value'] = pd.to_numeric(df['Value'], errors='coerce')
    df['Year']  = pd.to_numeric(df['Year'],  errors='coerce')
    df = df.dropna(subset=['Value', 'Year'])

    for col in ['Sex', 'Area']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    df['Year']   = df['Year'].astype(int)
    df['Decade'] = (df['Year'] // 10) * 10

    return df

df = load_data()

# --- TITLE ---
st.markdown('<div class="main-title">US Infant Mortality Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">BIOS 4505 / BMED 2400 · Data Source: UN Population Division</div>', unsafe_allow_html=True)

# --- SIDEBAR ---
st.sidebar.markdown("## 🎛️ Controls")

all_years   = sorted(df['Year'].unique())
all_decades = sorted(df['Decade'].unique())

selected_year   = st.sidebar.select_slider("Baby Chart — Year", options=all_years, value=all_years[-1])
selected_decade = st.sidebar.select_slider("Rose Chart — Decade", options=all_decades, value=all_decades[-1])

st.sidebar.markdown("---")
st.sidebar.markdown("**Color Guide**")
st.sidebar.markdown("🔵 = Male deaths &nbsp;&nbsp; 🩷 = Female deaths")
st.sidebar.markdown("Each pixel = 1% of infant deaths")

# ─────────────────────────────────────────────
# BABY SILHOUETTE — pixel map (10 cols × 14 rows = 140 cells, 100 "body" pixels)
# 1 = body pixel, 0 = empty
# ─────────────────────────────────────────────
BABY_MASK = np.array([
    # Row 0-1: head (cols 3-6)
    [0,0,0,1,1,1,1,0,0,0],
    [0,0,1,1,1,1,1,1,0,0],
    # Row 2: head bottom / neck
    [0,0,1,1,1,1,1,1,0,0],
    [0,0,0,1,1,1,1,0,0,0],
    # Row 3: neck
    [0,0,0,0,1,1,0,0,0,0],
    # Row 4-5: shoulders / arms start
    [0,1,1,1,1,1,1,1,1,0],
    [1,1,1,1,1,1,1,1,1,1],
    # Row 6-7: torso
    [0,1,1,1,1,1,1,1,1,0],
    [0,1,1,1,1,1,1,1,1,0],
    # Row 8: hips
    [0,0,1,1,1,1,1,1,0,0],
    # Row 9-10: thighs
    [0,0,1,1,0,0,1,1,0,0],
    [0,0,1,1,0,0,1,1,0,0],
    # Row 11-12: lower legs
    [0,0,1,1,0,0,1,1,0,0],
    [0,0,1,0,0,0,0,1,0,0],
])

def count_body_pixels(mask):
    return int(mask.sum())

TOTAL_BODY = count_body_pixels(BABY_MASK)   # should be ~100

# ─────────────────────────────────────────────
# LAYOUT
# ─────────────────────────────────────────────
col_left, col_right = st.columns([1, 1], gap="large")

# ══════════════════════════════════════════════
# LEFT: BABY PIXEL ART
# ══════════════════════════════════════════════
with col_left:
    st.markdown(f"### 👶 Sex Ratio — {selected_year}")
    st.markdown('<div class="section-label">Each pixel = 1% of infant deaths · Blue = Male · Pink = Female</div>', unsafe_allow_html=True)

    w_data = df[
        (df['Year'] == selected_year) &
        (df['Area'].str.contains('Total', case=False, na=False))
    ]

    male_val   = w_data[w_data['Sex'].str.contains('Male',   case=False, na=False)]['Value'].sum()
    female_val = w_data[w_data['Sex'].str.contains('Female', case=False, na=False)]['Value'].sum()
    total_val  = male_val + female_val

    if total_val == 0:
        # Fallback: try without Area filter
        w_data2    = df[df['Year'] == selected_year]
        male_val   = w_data2[w_data2['Sex'].str.contains('Male',   case=False, na=False)]['Value'].sum()
        female_val = w_data2[w_data2['Sex'].str.contains('Female', case=False, na=False)]['Value'].sum()
        total_val  = male_val + female_val

    if total_val > 0:
        male_pct   = male_val / total_val          # fraction [0,1]
        male_pixels = round(male_pct * TOTAL_BODY) # how many body pixels are blue

        rows, cols = BABY_MASK.shape
        fig, ax = plt.subplots(figsize=(5, 7))
        fig.patch.set_facecolor('#f8f9ff')
        ax.set_facecolor('#f8f9ff')

        pixel_idx = 0   # counts body pixels filled so far
        for r in range(rows):
            for c in range(cols):
                if BABY_MASK[r, c] == 1:
                    color = '#3b82f6' if pixel_idx < male_pixels else '#f472b6'
                    rect = mpatches.FancyBboxPatch(
                        (c, rows - r - 1), 0.85, 0.85,
                        boxstyle="round,pad=0.08",
                        facecolor=color,
                        edgecolor='white',
                        linewidth=1.5
                    )
                    ax.add_patch(rect)
                    pixel_idx += 1
                else:
                    # faint ghost pixel for context
                    rect = mpatches.FancyBboxPatch(
                        (c, rows - r - 1), 0.85, 0.85,
                        boxstyle="round,pad=0.08",
                        facecolor='#e5e7eb',
                        edgecolor='white',
                        linewidth=1,
                        alpha=0.35
                    )
                    ax.add_patch(rect)

        ax.set_xlim(-0.2, cols)
        ax.set_ylim(-0.2, rows)
        ax.set_aspect('equal')
        ax.axis('off')

        # Legend
        blue_patch  = mpatches.Patch(color='#3b82f6', label=f'Male  {male_pct*100:.1f}%')
        pink_patch  = mpatches.Patch(color='#f472b6', label=f'Female {(1-male_pct)*100:.1f}%')
        ax.legend(handles=[blue_patch, pink_patch], loc='lower center',
                  bbox_to_anchor=(0.5, -0.04), ncol=2,
                  fontsize=11, frameon=False)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        # Metric strip
        c1, c2, c3 = st.columns(3)
        c1.metric("Male Deaths",   f"{int(male_val):,}")
        c2.metric("Female Deaths", f"{int(female_val):,}")
        c3.metric("Male %",        f"{male_pct*100:.1f}%")
    else:
        st.warning(f"No male/female data found for {selected_year}. Check your CSV column names.")
        st.write("**Available Sex values:**", df['Sex'].unique()[:10])
        st.write("**Available Area values:**", df['Area'].unique()[:10])

# ══════════════════════════════════════════════
# RIGHT: ROSE / POLAR CHART — Urban vs Rural
# ══════════════════════════════════════════════
with col_right:
    st.markdown(f"### 🌹 Urban vs Rural — {selected_decade}s")
    st.markdown('<div class="section-label">Total infant deaths · Both sexes combined · Polar bar chart</div>', unsafe_allow_html=True)

    rose_df = df[
        (df['Decade'] == selected_decade) &
        (df['Area'].str.isin(['Urban', 'Rural'])) &
        (df['Sex'].str.contains('Both|Total', case=False, na=False))
    ]

    # Fallback: if 'Both' not found try aggregating all sexes
    if rose_df.empty:
        rose_df = df[
            (df['Decade'] == selected_decade) &
            (df['Area'].str.isin(['Urban', 'Rural']))
        ]

    if rose_df.empty:
        st.warning(f"No Urban/Rural data found for the {selected_decade}s.")
        st.write("**Available Area values:**", df['Area'].unique())
        st.write("**Available Decade values:**", df['Decade'].unique())
    else:
        u_val = rose_df[rose_df['Area'] == 'Urban']['Value'].sum()
        r_val = rose_df[rose_df['Area'] == 'Rural']['Value'].sum()

        # Build year-by-year trend for all decades for a richer rose
        trend_df = df[
            df['Area'].str.isin(['Urban', 'Rural']) &
            df['Sex'].str.contains('Both|Total', case=False, na=False)
        ].groupby(['Decade', 'Area'])['Value'].sum().reset_index()

        if trend_df.empty:
            trend_df = df[df['Area'].str.isin(['Urban', 'Rural'])
            ].groupby(['Decade', 'Area'])['Value'].sum().reset_index()

        decades_all = sorted(trend_df['Decade'].unique())
        n = len(decades_all)

        if n == 0:
            st.warning("Not enough data for rose chart.")
        else:
            # Each decade gets an angular "slice"; Urban & Rural side-by-side within
            theta_step = 2 * np.pi / n
            fig_r, ax_r = plt.subplots(figsize=(6, 6), subplot_kw={'projection': 'polar'})
            fig_r.patch.set_facecolor('#f8f9ff')
            ax_r.set_facecolor('#f8f9ff')

            bar_width = theta_step * 0.38

            for i, dec in enumerate(decades_all):
                theta_center = i * theta_step
                u = trend_df[(trend_df['Decade'] == dec) & (trend_df['Area'] == 'Urban')]['Value'].sum()
                r = trend_df[(trend_df['Decade'] == dec) & (trend_df['Area'] == 'Rural')]['Value'].sum()

                highlight = (dec == selected_decade)
                u_alpha = 1.0 if highlight else 0.45
                r_alpha = 1.0 if highlight else 0.45
                u_edge  = '#1d4ed8' if highlight else 'white'
                r_edge  = '#6b7280' if highlight else 'white'

                ax_r.bar(theta_center - bar_width/2, u, width=bar_width,
                         color='#3b82f6', alpha=u_alpha, edgecolor=u_edge, linewidth=1.5)
                ax_r.bar(theta_center + bar_width/2, r, width=bar_width,
                         color='#9ca3af', alpha=r_alpha, edgecolor=r_edge, linewidth=1.5)

            # Tick labels = decade names
            ax_r.set_xticks([i * theta_step for i in range(n)])
            ax_r.set_xticklabels([f"'{str(d)[2:]}s" for d in decades_all], fontsize=9)
            ax_r.set_yticklabels([])
            ax_r.set_theta_zero_location('N')
            ax_r.set_theta_direction(-1)
            ax_r.spines['polar'].set_visible(False)
            ax_r.grid(color='#e5e7eb', linewidth=0.6)

            # Legend
            blue_p = mpatches.Patch(color='#3b82f6', label='Urban')
            grey_p = mpatches.Patch(color='#9ca3af', label='Rural')
            ax_r.legend(handles=[blue_p, grey_p], loc='lower center',
                        bbox_to_anchor=(0.5, -0.12), ncol=2,
                        fontsize=11, frameon=False)

            plt.tight_layout()
            st.pyplot(fig_r)
            plt.close(fig_r)

            # Summary callout
            if u_val > 0 or r_val > 0:
                gap = r_val - u_val
                c1, c2 = st.columns(2)
                c1.metric("Urban Deaths", f"{int(u_val):,}")
                c2.metric("Rural Deaths", f"{int(r_val):,}")

                if gap > 0:
                    st.error(f"🚨 **Rural Penalty:** {int(gap):,} more deaths in rural areas during the {selected_decade}s.")
                elif gap < 0:
                    st.info(f"🏙️ **Urban Concentration:** {int(abs(gap)):,} more deaths recorded in urban areas during the {selected_decade}s.")
                else:
                    st.success("⚖️ Urban and rural deaths were equal this decade.")

st.divider()
st.caption("BIOS 4505 / BMED 2400 · US Infant Mortality · UN Population Division")
