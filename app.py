import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# --- 1. SETUP & DATA LOADING ---
st.set_page_config(page_title="US Infant Mortality BMED Study", layout="wide")

@st.cache_data
def load_and_clean_data():
    # Ensure this filename exactly matches the one in your GitHub repository
    file_name = "UNdata_Export_20260417_223711907.csv"
    
    if not os.path.exists(file_name):
        st.error(f"File {file_name} not found in the repository! Check the filename case sensitivity.")
        st.stop()
        
    df = pd.read_csv(file_name)
    # Filter for USA
    df = df[df['Country or Area'] == 'United States of America']
    # Clean 'Value' column
    df['Value'] = pd.to_numeric(df['Value'], errors='coerce')
    df = df.dropna(subset=['Value', 'Year', 'Sex', 'Area'])
    # Aggregation readiness
    df['Year'] = df['Year'].astype(int)
    df['Decade'] = (df['Year'] // 10) * 10
    return df

try:
    df = load_and_clean_data()
except Exception as e:
    st.error(f"Data loading failed: {e}")
    st.stop()

st.title("👶 US Infant Mortality: A Longitudinal Statistical Analysis")
st.markdown("""
<style>
.big-font { font-size:20px !important; font-weight:bold;}
</style>
""", unsafe_allow_html=True)

# --- SIDEBAR: TWO SLIDERS ---
st.sidebar.header("📊 Interactive Controls")

# Slider 1: Year for the Waffle Chart
all_years = sorted(df['Year'].unique())
selected_year = st.sidebar.select_slider(
    "1. Select Year (Sex Ratio)",
    options=all_years,
    value=all_years[0]
)

# Slider 2: Decade Range for the Rose Chart
all_decades = sorted(df['Decade'].unique())
selected_decade_range = st.sidebar.slider(
    "2. Select Decade Range (Urban/Rural)",
    min_value=int(all_decades[0]),
    max_value=int(all_decades[-1]),
    value=(int(all_decades[0]), int(all_decades[-1])),
    step=10
)

st.sidebar.divider()
st.sidebar.info("""
**Research Questions:**
* **Q1:** Has the ratio of deaths between sexes remained a biological constant?
* **Q2:** Are rural areas being left behind in the decline of mortality?
""")

# --- LAYOUT ---
col_waffle, col_rose = st.columns(2)

# --- COLUMN 1: THE BABY WAFFLE (QUESTION 1) ---
with col_waffle:
    st.header(f"Q1: Biological Sex Ratio ({selected_year})")
    
    # Filter for selected year and TOTAL area
    w_data = df[(df['Year'] == selected_year) & (df['Area'] == 'Total')]
    m_deaths = w_data[w_data['Sex'] == 'Male']['Value'].sum()
    f_deaths = w_data[w_data['Sex'] == 'Female']['Value'].sum()
    
    if (m_deaths + f_deaths) > 0:
        male_pct = int((m_deaths / (m_deaths + f_deaths)) * 100)
        
        # Visualize Waffle Chart
        fig_w, ax_w = plt.subplots(figsize=(6, 6))
        x, y = np.meshgrid(np.arange(10), np.arange(10))
        x, y = x.flatten(), y.flatten()
        
        # Colors: Blue=Male (#3498db), Pink=Female (#ff69b4)
        sex_colors = ['#3498db' if i < male_pct else '#ff69b4' for i in range(100)]
        
        ax_w.scatter(x, y, c=sex_colors, marker='s', s=450, edgecolors='white', linewidth=0.5)
        ax_w.axis('off')
        ax_w.set_aspect('equal')
        
        # Adding a title inside the plot area for the "Baby" context
        ax_w.text(4.5, 10, "🍼 Biological Vulnerability Check", ha='center', fontsize=14, weight='bold')
        
        st.pyplot(fig_w)
        st.markdown(f'<p class="big-font">♂️ Male: <span style="color:#3498db">{male_pct}%</span> | ♀️ Female: <span style="color:#ff69b4">{100-male_pct}%</span></p>', unsafe_allow_html=True)
    else:
        st.warning(f"No sex-specific data found for {selected_year}.")

# --- COLUMN 2: THE URBAN-RURAL ROSE (QUESTION 2) ---
with col_rose:
    st.header(f"Q2: Urban vs. Rural Gap")
    
    # Process Rose Data within selected decade range
    min_d, max_d = selected_decade_range
    rose_df = df[(df['Decade'] >= min_d) & (df['Decade'] <= max_d)]
    
    # Filter strictly for Urban/Rural areas and exclude "Both Sexes" to avoid double-counts
    gap_data = rose_df[(rose_df['Area'].isin(['Urban', 'Rural'])) & (rose_df['Sex'] != 'Both Sexes')]
    
    if gap_data.empty:
        st.warning("No Urban/Rural data available for this range.")
    else:
        pivot = gap_data.groupby(['Decade', 'Area'])['Value'].sum().unstack().dropna()
        
        labels = [f"{int(d)}s" for d in pivot.index]
        angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
        width = (2 * np.pi / len(labels)) * 0.7 
        
        fig_r, ax_r = plt.subplots(figsize=(6, 6), subplot_kw={'projection': 'polar'})
        
        # Plot Rural (Background, larger volume)
        ax_r.bar(angles, pivot['Rural'], width=width, color='lightgray', alpha=0.5, label='Rural deaths', edgecolor='black')
        # Plot Urban (Foreground, smaller volume)
        ax_r.bar(angles, pivot['Urban'], width=width, color='#3498db', alpha=0.7, label='Urban deaths', edgecolor='black')
        
        ax_r.set_theta_zero_location('N')
        ax_r.set_theta_direction(-1)
        ax_r.set_xticks(angles)
        ax_r.set_xticklabels(labels)
        ax_r.set_yticklabels([]) # Hide radius numbers for a cleaner "clock" look
        
        plt.legend(loc='lower right', bbox_to_anchor=(1.3, 0.1))
        st.pyplot(fig_r)
        
        st.markdown("**Observation:** Does the blue (Urban) wedge shrink faster than the gray (Rural) wedge over time?")

st.divider()
st.write("Source: UN Data - Infant deaths by sex and urban/rural residence.")
