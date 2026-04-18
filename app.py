import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import numpy as np
import os

# ── CONFIG ──────────────────────────────────────────────────────────────────
st.set_page_config(layout="wide", page_title="US Infant Mortality | BMED Study")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700&family=Source+Sans+3:wght@300;400;600&display=swap');

html, body, [class*="css"] { font-family: 'Source Sans 3', sans-serif; }

.main-title {
    font-family: 'Playfair Display', serif;
    font-size: 2.6rem; color: #0f172a;
    text-align: center; margin-bottom: 0.15rem; letter-spacing: -0.01em;
}
.subtitle {
    font-size: 0.82rem; color: #94a3b8; text-align: center;
    margin-bottom: 2.5rem; text-transform: uppercase; letter-spacing: 0.12em;
}
.chart-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.35rem; color: #1e293b; margin-bottom: 0.15rem;
}
.chart-sub {
    font-size: 0.75rem; color: #94a3b8; text-transform: uppercase;
    letter-spacing: 0.08em; margin-bottom: 0.8rem;
}
</style>
""", unsafe_allow_html=True)

# ── DATA LOADING ─────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    file_name = "UNdata_Export_20260417_223711907.csv"
    if not os.path.exists(file_name):
        st.error(f"File `{file_name}` not found.")
        st.stop()

    df = pd.read_csv(file_name, low_memory=False)
    df.columns = [str(c).strip() for c in df.columns]

    # Rename based on known UN export format
    rename = {
        'Country or Area': 'Country',
        'Year':            'Year',
        'Area':            'Area',
        'Sex':             'Sex',
        'Value':           'Value',
    }
    # Keep only columns we need
    df = df.rename(columns=rename)
    keep = [c for c in ['Country','Year','Area','Sex','Value'] if c in df.columns]
    df = df[keep]

    df = df[df['Country'].astype(str).str.contains('United States', na=False)]
    df['Value'] = pd.to_numeric(df['Value'], errors='coerce')
    df['Year']  = pd.to_numeric(df['Year'],  errors='coerce')
    df = df.dropna(subset=['Value', 'Year'])

    df['Sex']    = df['Sex'].astype(str).str.strip()
    df['Area']   = df['Area'].astype(str).str.strip()
    df['Year']   = df['Year'].astype(int)
    df['Decade'] = (df['Year'] // 10) * 10

    return df

df = load_data()

# ── TITLE ─────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">US Infant Mortality</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle"> / BMED 2400 · UN Population Division</div>', unsafe_allow_html=True)

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
st.sidebar.markdown("## 🎛️ Controls")
all_years = sorted(df['Year'].unique())
selected_year = st.sidebar.select_slider("Select Year (Baby Chart)", options=all_years, value=all_years[-1])

st.sidebar.markdown("---")
st.sidebar.markdown("**Blue pixels** = % male deaths")
st.sidebar.markdown("**Pink pixels** = % female deaths")
st.sidebar.markdown("Each pixel = 1 out of 100 infant deaths")

with st.sidebar.expander("🔍 Debug: data check"):
    yr = df[df['Year'] == selected_year]
    st.write("Sex values:", sorted(df['Sex'].unique()))
    st.write("Area values:", sorted(df['Area'].unique()))
    st.dataframe(yr[['Year','Sex','Area','Value']].head(20))

# ── BABY MASK — exactly 100 body pixels ──────────────────────────────────────
BABY_MASK = np.array([
    [0,0,0,1,1,1,1,0,0,0],   # 4   HEAD
    [0,0,1,1,1,1,1,1,0,0],   # 6
    [0,0,1,1,1,1,1,1,0,0],   # 6
    [0,0,0,1,1,1,1,0,0,0],   # 4   → 20
    [0,0,0,0,1,1,0,0,0,0],   # 2   NECK → 22
    [0,1,1,1,1,1,1,1,1,0],   # 8   SHOULDERS
    [1,1,1,1,1,1,1,1,1,1],   # 10
    [1,1,1,1,1,1,1,1,1,1],   # 10  → 50
    [0,1,1,1,1,1,1,1,1,0],   # 8   TORSO
    [0,1,1,1,1,1,1,1,1,0],   # 8
    [0,0,1,1,1,1,1,1,0,0],   # 6   → 72
    [0,0,1,1,0,0,1,1,0,0],   # 4   LEGS
    [0,0,1,1,0,0,1,1,0,0],   # 4
    [0,0,1,1,0,0,1,1,0,0],   # 4
    [0,0,1,1,0,0,1,1,0,0],   # 4
    [0,0,1,1,0,0,1,1,0,0],   # 4   → 92
    [0,1,1,1,0,0,1,1,1,0],   # 6   FEET
    [0,0,1,0,0,0,0,1,0,0],   # 2   → 100
])
TOTAL_BODY = int(BABY_MASK.sum())  # confirmed = 100

# ── LAYOUT: two columns ──────────────────────────────────────────────────────
col_baby, col_clock = st.columns([1, 1], gap="large")

# ════════════════════════════════════════════════════════════════════════════
# LEFT — BABY PIXEL CHART
# ════════════════════════════════════════════════════════════════════════════
with col_baby:
    st.markdown(f'<div class="chart-title">👶 Sex Ratio of Infant Deaths — {selected_year}</div>', unsafe_allow_html=True)
    st.markdown('<div class="chart-sub">Each pixel = 1% · Blue = Male · Pink = Female</div>', unsafe_allow_html=True)

    yr_df = df[df['Year'] == selected_year]

    def get_sex_vals(sub):
        m = sub[
            sub['Sex'].str.contains('Male', case=False, na=False) &
            ~sub['Sex'].str.contains('Female', case=False, na=False)
        ]['Value'].sum()
        f = sub[sub['Sex'].str.contains('Female', case=False, na=False)]['Value'].sum()
        return float(m), float(f)

    # Prefer Area == Total rows; fall back to everything
    total_df = yr_df[yr_df['Area'].str.contains('Total', case=False, na=False)]
    male_val, female_val = get_sex_vals(total_df)
    if (male_val + female_val) == 0:
        male_val, female_val = get_sex_vals(yr_df)

    total_val = male_val + female_val

    if total_val > 0:
        male_pct    = male_val / total_val
        male_pixels = round(male_pct * TOTAL_BODY)

        ROWS, COLS = BABY_MASK.shape
        fig, ax = plt.subplots(figsize=(5, 8))
        fig.patch.set_facecolor('#f8fafc')
        ax.set_facecolor('#f8fafc')

        pixel_idx = 0
        for r in range(ROWS):
            for c in range(COLS):
                if BABY_MASK[r, c] == 1:
                    color = '#3b82f6' if pixel_idx < male_pixels else '#f472b6'
                    rect = mpatches.FancyBboxPatch(
                        (c, ROWS - r - 1), 0.84, 0.84,
                        boxstyle="round,pad=0.09",
                        facecolor=color, edgecolor='white', linewidth=1.8,
                        zorder=3
                    )
                    ax.add_patch(rect)
                    pixel_idx += 1
                else:
                    rect = mpatches.FancyBboxPatch(
                        (c, ROWS - r - 1), 0.84, 0.84,
                        boxstyle="round,pad=0.09",
                        facecolor='#cbd5e1', edgecolor='white',
                        linewidth=1, alpha=0.25, zorder=2
                    )
                    ax.add_patch(rect)

        ax.set_xlim(-0.4, COLS + 0.3)
        ax.set_ylim(-0.6, ROWS + 0.4)
        ax.set_aspect('equal')
        ax.axis('off')

        bp = mpatches.Patch(color='#3b82f6', label=f'Male  {male_pct*100:.1f}%  ({int(male_val):,})')
        pp = mpatches.Patch(color='#f472b6', label=f'Female  {(1-male_pct)*100:.1f}%  ({int(female_val):,})')
        ax.legend(handles=[bp, pp], loc='lower center',
                  bbox_to_anchor=(0.5, -0.02), ncol=2, fontsize=10.5,
                  frameon=False, handlelength=1.2)

        plt.tight_layout(pad=0.5)
        st.pyplot(fig)
        plt.close(fig)

        c1, c2, c3 = st.columns(3)
        c1.metric("Male Deaths",   f"{int(male_val):,}")
        c2.metric("Female Deaths", f"{int(female_val):,}")
        c3.metric("Male %",        f"{male_pct*100:.1f}%")
    else:
        st.warning(f"No male/female data for {selected_year}.")
        st.dataframe(yr_df[['Sex','Area','Value']].head(20))

# ════════════════════════════════════════════════════════════════════════════
# RIGHT — CLOCK-STYLE ROSE CHART (Urban vs Rural, all decades)
# ════════════════════════════════════════════════════════════════════════════
with col_clock:
    st.markdown('<div class="chart-title">🕐 Urban vs Rural Deaths — All Decades</div>', unsafe_allow_html=True)
    st.markdown('<div class="chart-sub">Clock layout · Each hour = one decade · Blue = Urban · Grey = Rural · Both Sexes</div>', unsafe_allow_html=True)

    # Get Urban/Rural rows — prefer "Both Sexes", fall back to all
    ur_df = df[df['Area'].isin(['Urban', 'Rural'])]
    rose_df = ur_df[ur_df['Sex'].str.contains('Both|Total', case=False, na=False)]
    if rose_df.empty:
        rose_df = ur_df

    if rose_df.empty:
        st.warning("No Urban/Rural data found.")
        st.write("Available Area values:", sorted(df['Area'].unique()))
    else:
        trend = (
            rose_df
            .groupby(['Decade', 'Area'])['Value']
            .sum()
            .reset_index()
        )

        decades = sorted(trend['Decade'].unique())
        n = len(decades)

        # ── Clock face setup ──────────────────────────────────────────────
        # Place decades like clock hours: 12 o'clock = earliest decade,
        # going clockwise. Each decade gets an angular slot of 2π/n.
        # Two bars per slot: Urban (blue, inner-left) & Rural (grey, inner-right).

        fig_c, ax_c = plt.subplots(figsize=(7, 7), subplot_kw={'projection': 'polar'})
        fig_c.patch.set_facecolor('#f8fafc')
        ax_c.set_facecolor('#f8fafc')

        # Clock: zero at top, clockwise
        ax_c.set_theta_zero_location('N')
        ax_c.set_theta_direction(-1)

        theta_step = 2 * np.pi / n
        bar_w      = theta_step * 0.36   # width of each individual bar

        u_max = trend[trend['Area'] == 'Urban']['Value'].max()
        r_max = trend[trend['Area'] == 'Rural']['Value'].max()
        global_max = max(u_max, r_max) if (u_max > 0 or r_max > 0) else 1

        for i, dec in enumerate(decades):
            theta_center = i * theta_step

            u = trend[(trend['Decade'] == dec) & (trend['Area'] == 'Urban')]['Value'].sum()
            r = trend[(trend['Decade'] == dec) & (trend['Area'] == 'Rural')]['Value'].sum()

            # Urban bar (left of center)
            ax_c.bar(
                theta_center - bar_w * 0.55, u,
                width=bar_w, color='#3b82f6', alpha=0.88,
                edgecolor='white', linewidth=0.8, bottom=0
            )
            # Rural bar (right of center)
            ax_c.bar(
                theta_center + bar_w * 0.55, r,
                width=bar_w, color='#64748b', alpha=0.75,
                edgecolor='white', linewidth=0.8, bottom=0
            )

        # ── Clock-face tick labels (decade = "hour") ──────────────────────
        tick_angles = [i * theta_step for i in range(n)]
        ax_c.set_xticks(tick_angles)
        ax_c.set_xticklabels(
            [f"'{str(d)[2:]}s" for d in decades],
            fontsize=8.5, color='#334155', fontweight='600'
        )

        # Hide radial gridlines, keep angular lines subtle
        ax_c.set_yticklabels([])
        ax_c.yaxis.grid(True, color='#e2e8f0', linewidth=0.6, linestyle='--')
        ax_c.xaxis.grid(True, color='#e2e8f0', linewidth=0.4)
        ax_c.spines['polar'].set_color('#e2e8f0')
        ax_c.spines['polar'].set_linewidth(1)

        # ── Clock centre dot ──────────────────────────────────────────────
        ax_c.plot(0, 0, 'o', color='#1e293b', markersize=6, zorder=10,
                  transform=ax_c.transData)

        # ── Legend ────────────────────────────────────────────────────────
        up = mpatches.Patch(color='#3b82f6', alpha=0.88, label='Urban')
        rp = mpatches.Patch(color='#64748b', alpha=0.75, label='Rural')
        ax_c.legend(
            handles=[up, rp],
            loc='lower center', bbox_to_anchor=(0.5, -0.12),
            ncol=2, fontsize=10.5, frameon=False
        )

        plt.tight_layout(pad=0.5)
        st.pyplot(fig_c)
        plt.close(fig_c)

        # ── Data note ─────────────────────────────────────────────────────
        st.caption(
            "⚠️ Urban/Rural breakdown only available for select years in this dataset "
            "(1960–1968 and 2010–2019). Decades with no Urban/Rural data show as empty sectors."
        )

        # ── Summary table ─────────────────────────────────────────────────
        with st.expander("📊 View decade totals"):
            summary = trend.pivot(index='Decade', columns='Area', values='Value').fillna(0).astype(int)
            summary.index = [f"'{str(d)[2:]}s" for d in summary.index]
            if 'Urban' in summary.columns and 'Rural' in summary.columns:
                summary['Rural Penalty'] = summary['Rural'] - summary['Urban']
            st.dataframe(summary, use_container_width=True)

st.divider()
st.caption("BIOS 4505 / BMED 2400 · US Infant Mortality · UN Population Division")
