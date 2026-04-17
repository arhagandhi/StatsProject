import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib.path as mpath
import matplotlib.lines as mlines

# --- 1. SETUP & DATA LOADING ---
st.set_page_config(page_title="US Infant Mortality BMED Study", layout="wide")

@st.cache_data
def load_and_clean_data():
    # Use the filename as it exists in your GitHub repo
    file_name = "UNdata_Export_20260417_223711907.csv"
    if not os.path.exists(file_name):
        st.error(f"File {file_name} not found in the repository!")
        st.stop()
        
    df = pd.read_csv(file_name)
    # 1. Filter for United States of America
    df = df[df['Country or Area'] == 'United States of America']
    # 2. Clean 'Value' column
    df['Value'] = pd.to_numeric(df['Value'], errors='coerce')
    df = df.dropna(subset=['Value', 'Year', 'Sex', 'Area'])
    # 3. Aggregation readiness
    df['Year'] = df['Year'].astype(int)
    df['Decade'] = (df['Year'] // 10) * 10
    return df

try:
    df = load_and_clean_data()
except Exception as e:
    st.error(f"Data loading failed: {e}")
    st.stop()

st.title("👶 US Infant Mortality: A Longitudinal Statistical Analysis (1948–Present)")
st.markdown("""
<style>
.big-font { font-size:20px !important; font-weight:bold;}
</style>
""", unsafe_allow_stdio=True)

# --- SIDEBAR: TWO SLIDERS ---
st.sidebar.header("📊 Interactive Controls")

# Slider 1: Waffle Year
all_years = sorted(df['Year'].unique())
selected_year = st.sidebar.select_slider(
    "1. Select Year for Sex Ratio (Waffle Chart)",
    options=all_years,
    value=all_years[0]
)

# Slider 2: Rose Decade Range
all_decades = sorted(df['Decade'].unique())
selected_decade_range = st.sidebar.slider(
    "2. Select Decade Range for Urban/Rural Gap (Rose Chart)",
    min_value=int(all_decades[0]),
    max_value=int(all_decades[-1]),
    value=(int(all_decades[0]), int(all_decades[-1])),
    step=10
)

st.sidebar.markdown("---")
st.sidebar.write("### Study Importance")
st.sidebar.write("""
1. **Sex Ratio:** If the biological male disadvantage is narrowing, it proves neonatal care technology (like ventilation) is overcoming natural vulnerability.
2. **Urban-Rural Gap:** If total deaths are falling but the rural *share* is not, it indicates a geography-based failure in healthcare distribution, not medical technology.
""")


# --- LAYOUT ---
col_waffle, col_rose = st.columns(2)

# --- COLUMN 1: THE "BABY" WAFFLE (QUESTION 1) ---
with col_waffle:
    st.header(f"Q1: The Biological Sex Ratio ({selected_year})")
    
    # Process Waffle Data
    # Filter for selected year and TOTAL area (to isolate sex ratio)
    w_data = df[(df['Year'] == selected_year) & (df['Area'] == 'Total')]
    m_deaths = w_data[w_data['Sex'] == 'Male']['Value'].sum()
    f_deaths = w_data[w_data['Sex'] == 'Female']['Value'].sum()
    
    if (m_deaths + f_deaths) > 0:
        male_pct = int((m_deaths / (m_deaths + f_deaths)) * 100)
        
        # Visualize Waffle Chart
        fig_w, ax_w = plt.subplots(figsize=(6, 7))
        x, y = np.meshgrid(np.arange(10), np.arange(10))
        x, y = x.flatten(), y.flatten()
        
        # Defining Sex Colors: Blue=Male, Pink=Female
        # We fill the male percentage first.
        sex_colors = ['#3498db' if i < male_pct else '#ff69b4' for i in range(100)]
        
        # Use a square marker but add a static baby graphic above.
        # Plotting the 10x10 waffle grid
        ax_w.scatter(x, y, c=sex_colors, marker='s', s=450, edgecolors='white', linewidth=0.5)
        
        # Aesthetics: Remove axes, set title
        ax_w.axis('off')
        ax_w.set_aspect('equal')
        
        # We add the 'Baby' context using an annotation above the chart
        ax_w.text(4.5, 10.5, "👶 Biological Constant?", ha='center', va='bottom', fontsize=18, color='#333333')
        
        st.pyplot(fig_w)
        
        st.markdown(f'<p class="big-font">♂️ Male Mortality Percentage: <span style="color:#3498db">{male_pct}%</span> | ♀️ Female: <span style="color:#ff69b4">{100-male_pct}%</span></p>', unsafe_allow_stdio=True)

    else:
        st.warning(f"No sex-specific data found for {selected_year}. Try another year!")

# --- COLUMN 2: THE urban-RURAL 'CLOCK' (QUESTION 2) ---
with col_rose:
    st.header(f"Q2: The Urban-Rural Gap ({selected_decade_range[0]}s - {selected_decade_range[1]}s)")
    
    # Process Rose Data
    min_d, max_d = selected_decade_range
    rose_df = df[(df['Decade'] >= min_d) & (df['Decade'] <= max_d)]
    
    # We must filter out "Total" area, only looking at 'Urban' vs 'Rural'
    # And must exclude "Both Sexes" to avoid double-counting M+F.
    gap_data = rose_df[ (df['Area'].isin(['Urban', 'Rural'])) & (df['Sex'] != 'Both Sexes') ]
    
    if gap_data.empty:
        st.warning("No Urban/Rural disaggregated data available for this range.")
    else:
        # Group and pivot to get Urban/Rural columns per decade
        # We sum M+F deaths within each Area/Decade.
        pivot = gap_data.groupby(['Decade', 'Area'])['Value'].sum().unstack().dropna()
        
        labels = [f"{int(d)}s" for d in pivot.index]
        num_labels = len(labels)
        angles = np.linspace(0, 2 * np.pi, num_labels, endpoint=False)
        width = (2 * np.pi / num_labels) * 0.7  # Use thinner wedges to create the Coxcomb look
        
        fig_r, ax_r = plt.subplots(figsize=(7, 7), subplot_kw={'projection': 'polar'})
        
        # Coxcomb logic: We are plotting 'Value' as the width and the *area* of the wedge.
        # This requires adjusting the radius for each bar if we were doing a pure Coxcomb,
        # but the overlap method works well here for highlighting the 'gap'.
        
        # Plot Rural (Background, larger volume)
        ax_r.bar(angles, pivot['Rural'], width=width, color='gray', alpha=0.3, label='Rural deaths', edgecolor='black')
        # Plot Urban (Foreground, smaller volume)
        ax_r.bar(angles, pivot['Urban'], width=width, color='blue', alpha=0.6, label='Urban deaths', edgecolor='black')
        
        # Formatting the Polar Plot
        ax_r.set_theta_zero_location('N') # 12 o'clock start
        ax_r.set_theta_direction(-1)     # Clockwise progression
        ax_r.set_xticks(angles)
        ax_r.set_xticklabels(labels, fontsize=12)
        ax_r.tick_params(axis='y', colors='black')
        
        # Adding a visual guide for the scale
        y_ticks = ax_r.get_yticks()
        ax_r.set_rlabel_position(180) # 6 o'clock labels
        
        plt.title("Volume of Infant Deaths: Rural Penalty", size=18, pad=35)
        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        
        st.pyplot(fig_r)
        
        with st.expander("View Rose Chart Data Table"):
            st.dataframe(pivot)

st.divider()
st.write("This dashboard is powered by the [UN Data Division's Infant Mortality dataset](http://data.un.org/Data.aspx?q=infant&d=POP&f=tableCode%3a9).")
