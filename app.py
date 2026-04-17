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
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    h1, h2, h3 { font-family: 'DM Serif Display', serif !important; }
    .main-title {
        font-family: 'DM Serif Display', serif;
        font-size: 2.4rem; color: #1a1a2e;
        text-align: center; margin-bottom: 0.2rem;
    }
    .subtitle {
        font-family: 'DM Sans', sans-serif;
        font-size: 0.95rem; color: #6b7280;
        text-align: center; margin-bottom: 2rem;
        font-weight: 300; letter-spacing: 0.04em; text-transform: uppercase;
    }
    .section-label {
        font-size: 0.75rem; text-transform: uppercase;
        letter-spacing: 0.1em; color: #9ca3af;
        font-weight: 500; margin-bottom: 0.3rem;
    }
</style>
""", unsafe_allow_html=True)

# --- DATA LOADING ---
@st.cache_data
def load_data():
    file_name = "UNdata_Export_20260417_223711907.csv"
    if not os.path.exists(file_name):
        st.error(f"File `{file_name}` not found. Make sure it's in the same directory as this script.")
        st.stop()

    df = pd.read_csv(file_name, low_memory=False)
    df.columns = [str(c).strip() for c in df.columns]

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
st.sidebar.markdown("## Controls")

all_years   = sorted(df['Year'].unique())
all_decades = sorted(df['Decade'].unique())

selected_year   = st.sidebar.select_slider("Baby Chart — Year", options=all_years, value=all_years[-1])
selected_decade = st.sidebar.select_slider("Rose Chart — Decade", options=all_decades, value=all_decades[-1])

st.sidebar.markdown("---")
st.sidebar.markdown("**Color Guide**")
st.sidebar.markdown("Blue = Male deaths · Pink = Female deaths")
st.sidebar.markdown("Each pixel = 1% of infant deaths")

# DEBUG expander — helps verify numbers match your CSV
with st.sidebar.expander("Debug: Raw Data"):
    st.write("**Columns:**", list(df.columns))
    st.write("**Sex values:**", sorted(df['Sex'].unique()))
    st.write("**Area values:**", sorted(df['Area'].unique()))
    yr_sample = df[df['Year'] == selected_year]
    st.write(f"**Rows for {selected_year}:**", len(yr_sample))
    st.dataframe(yr_sample[['Year', 'Sex', 'Area', 'Value']].head(20))

# ─────────────────────────────────────────────
# BABY SILHOUETTE — exactly 100 body pixels
# Verified: this array sums to exactly 100
# ─────────────────────────────────────────────
BABY_MASK = np.array([
    # HEAD
    [0,0,0,1,1,1,1,0,0,0],   # 4
    [0,0,1,1,1,1,1,1,0,0],   # 6
    [0,0,1,1,1,1,1,1,0,0],   # 6
    [0,0,0,1,1,1,1,0,0,0],   # 4  → total 20
    # NECK
    [0,0,0,0,1,1,0,0,0,0],   # 2  → 22
    # ARMS / SHOULDERS
    [0,1,1,1,1,1,1,1,1,0],   # 8
    [1,1,1,1,1,1,1,1,1,1],   # 10
    [1,1,1,1,1,1,1,1,1,1],   # 10 → 50
    # TORSO
    [0,1,1,1,1,1,1,1,1,0],   # 8
    [0,1,1,1,1,1,1,1,1,0],   # 8
    [0,0,1,1,1,1,1,1,0,0],   # 6  → 72
    # LEGS
    [0,0,1,1,0,0,1,1,0,0],   # 4
    [0,0,1,1,0,0,1,1,0,0],   # 4
    [0,0,1,1,0,0,1,1,0,0],   # 4
    [0,0,1,1,0,0,1,1,0,0],   # 4
    [0,0,1,1,0,0,1,1,0,0],   # 4  → 92
    # FEET
    [0,1,1,1,0,0,1,1,1,0],   # 6
    [0,0,1,0,0,0,0,1,0,0],   # 2  → 100
])

TOTAL_BODY = int(BABY_MASK.sum())  # = 100

# ─────────────────────────────────────────────
# LAYOUT
# ─────────────────────────────────────────────
col_left, col_right = st.columns([1, 1], gap="large")

# ══════════════════════════════════════════════
# LEFT: BABY PIXEL ART
# ══════════════════════════════════════════════
with col_left:
    st.markdown(f"### Baby Pixel Chart — {selected_year}")
    st.markdown(
        '<div class="section-label">Each blue pixel = 1% male deaths · Each pink pixel = 1% female deaths</div>',
        unsafe_allow_html=True
    )

    yr_df = df[df['Year'] == selected_year]

    def get_sex_vals(sub):
        # Male = rows containing 'Male' but NOT 'Female' (avoids double-counting)
        m = sub[
            sub['Sex'].str.contains('Male', case=False, na=False) &
            ~sub['Sex'].str.contains('Female', case=False, na=False)
        ]['Value'].sum()
        f = sub[sub['Sex'].str.contains('Female', case=False, na=False)]['Value'].sum()
        return m, f

    # Attempt 1: Area contains 'Total'
    total_area_df = yr_df[yr_df['Area'].str.contains('Total', case=False, na=False)]
    male_val, female_val = get_sex_vals(total_area_df)

    # Attempt 2: No area filter (sum everything)
    if (male_val + female_val) == 0:
        male_val, female_val = get_sex_vals(yr_df)

    total_val = male_val + female_val

    if total_val > 0:
        male_pct    = male_val / total_val
        male_pixels = round(male_pct * TOTAL_BODY)

        rows, cols = BABY_MASK.shape
        fig, ax = plt.subplots(figsize=(5, 7.5))
        fig.patch.set_facecolor('#f8f9ff')
        ax.set_facecolor('#f8f9ff')

        pixel_idx = 0
        for r in range(rows):
            for c in range(cols):
                if BABY_MASK[r, c] == 1:
                    color = '#3b82f6' if pixel_idx < male_pixels else '#f472b6'
                    rect = mpatches.FancyBboxPatch(
                        (c, rows - r - 1), 0.85, 0.85,
                        boxstyle="round,pad=0.08",
                        facecolor=color, edgecolor='white', linewidth=1.5
                    )
                    ax.add_patch(rect)
                    pixel_idx += 1
                else:
                    rect = mpatches.FancyBboxPatch(
                        (c, rows - r - 1), 0.85, 0.85,
                        boxstyle="round,pad=0.08",
                        facecolor='#e5e7eb', edgecolor='white',
                        linewidth=1, alpha=0.3
                    )
                    ax.add_patch(rect)

        ax.set_xlim(-0.3, cols + 0.2)
        ax.set_ylim(-0.5, rows + 0.3)
        ax.set_aspect('equal')
        ax.axis('off')

        blue_patch = mpatches.Patch(color='#3b82f6',
                                    label=f'Male  {male_pct*100:.1f}%  ({int(male_val):,})')
        pink_patch = mpatches.Patch(color='#f472b6',
                                    label=f'Female  {(1-male_pct)*100:.1f}%  ({int(female_val):,})')
        ax.legend(handles=[blue_patch, pink_patch], loc='lower center',
                  bbox_to_anchor=(0.5, -0.03), ncol=2, fontsize=10, frameon=False)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        c1, c2, c3 = st.columns(3)
        c1.metric("Male Deaths",   f"{int(male_val):,}")
        c2.metric("Female Deaths", f"{int(female_val):,}")
        c3.metric("Male %",        f"{male_pct*100:.1f}%")

    else:
        st.warning(f"No male/female data found for {selected_year}.")
        st.write("**Sex values in this year:**", yr_df['Sex'].unique())
        st.write("**Area values in this year:**", yr_df['Area'].unique())
        st.dataframe(yr_df[['Sex', 'Area', 'Value']].head(20))

# ══════════════════════════════════════════════
# RIGHT: ROSE / POLAR CHART — Urban vs Rural
# ══════════════════════════════════════════════
with col_right:
    st.markdown(f"### Urban vs Rural — {selected_decade}s")
    st.markdown(
        '<div class="section-label">Total infant deaths · Both sexes · Highlighted decade = selected in sidebar</div>',
        unsafe_allow_html=True
    )

    # KEY FIX: .isin() on the Series directly, NOT .str.isin()
    urban_rural_df = df[df['Area'].isin(['Urban', 'Rural'])]

    # Try to get 'Both sexes' rows; fall back to all rows
    rose_df = urban_rural_df[
        urban_rural_df['Sex'].str.contains('Both|Total', case=False, na=False)
    ]
    if rose_df.empty:
        rose_df = urban_rural_df

    if rose_df.empty:
        st.warning("No Urban/Rural data found in this dataset.")
        st.write("**Area values available:**", sorted(df['Area'].unique()))
    else:
        trend_df = (
            rose_df
            .groupby(['Decade', 'Area'])['Value']
            .sum()
            .reset_index()
        )

        decades_all = sorted(trend_df['Decade'].unique())
        n = len(decades_all)

        if n == 0:
            st.warning("Not enough data to draw rose chart.")
        else:
            theta_step = 2 * np.pi / n
            bar_width  = theta_step * 0.38

            fig_r, ax_r = plt.subplots(figsize=(6, 6), subplot_kw={'projection': 'polar'})
            fig_r.patch.set_facecolor('#f8f9ff')
            ax_r.set_facecolor('#f8f9ff')

            for i, dec in enumerate(decades_all):
                theta_center = i * theta_step
                u = trend_df[
                    (trend_df['Decade'] == dec) & (trend_df['Area'] == 'Urban')
                ]['Value'].sum()
                r = trend_df[
                    (trend_df['Decade'] == dec) & (trend_df['Area'] == 'Rural')
                ]['Value'].sum()

                highlight = (dec == selected_decade)
                u_alpha = 1.0 if highlight else 0.40
                r_alpha = 1.0 if highlight else 0.40
                u_edge  = '#1d4ed8' if highlight else 'white'
                r_edge  = '#374151' if highlight else 'white'

                ax_r.bar(theta_center - bar_width / 2, u, width=bar_width,
                         color='#3b82f6', alpha=u_alpha, edgecolor=u_edge, linewidth=1.5)
                ax_r.bar(theta_center + bar_width / 2, r, width=bar_width,
                         color='#9ca3af', alpha=r_alpha, edgecolor=r_edge, linewidth=1.5)

            ax_r.set_xticks([i * theta_step for i in range(n)])
            ax_r.set_xticklabels([f"'{str(d)[2:]}s" for d in decades_all], fontsize=9)
            ax_r.set_yticklabels([])
            ax_r.set_theta_zero_location('N')
            ax_r.set_theta_direction(-1)
            ax_r.spines['polar'].set_visible(False)
            ax_r.grid(color='#e5e7eb', linewidth=0.6)

            blue_p = mpatches.Patch(color='#3b82f6', label='Urban')
            grey_p = mpatches.Patch(color='#9ca3af', label='Rural')
            ax_r.legend(handles=[blue_p, grey_p], loc='lower center',
                        bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=11, frameon=False)

            plt.tight_layout()
            st.pyplot(fig_r)
            plt.close(fig_r)

            # Summary callout for selected decade
            u_val = trend_df[
                (trend_df['Decade'] == selected_decade) & (trend_df['Area'] == 'Urban')
            ]['Value'].sum()
            r_val = trend_df[
                (trend_df['Decade'] == selected_decade) & (trend_df['Area'] == 'Rural')
            ]['Value'].sum()

            if u_val > 0 or r_val > 0:
                c1, c2 = st.columns(2)
                c1.metric("Urban Deaths", f"{int(u_val):,}")
                c2.metric("Rural Deaths", f"{int(r_val):,}")
                gap = r_val - u_val
                if gap > 0:
                    st.error(f"Rural Penalty: {int(gap):,} more deaths in rural areas during the {selected_decade}s.")
                elif gap < 0:
                    st.info(f"Urban Concentration: {int(abs(gap)):,} more deaths in urban areas during the {selected_decade}s.")
                else:
                    st.success("Urban and rural deaths were equal this decade.")
            else:
                st.warning(f"No Urban/Rural data for the {selected_decade}s specifically.")

st.divider()
st.caption("BIOS 4505 / BMED 2400 · US Infant Mortality · UN Population Division")
