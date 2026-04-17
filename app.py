import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

st.set_page_config(layout="wide", page_title="US Infant Mortality Study")

@st.cache_data
def load_data():
    # Relative path for GitHub
    file_name = "UNdata_Export_20260417_223711907.csv"
    if not os.path.exists(file_name):
        st.error(f"File {file_name} not found!")
        st.stop()
    df = pd.read_csv(file_name)
    df = df[df['Country or Area'] == 'United States of America']
    df['Value'] = pd.to_numeric(df['Value'], errors='coerce')
    df = df.dropna(subset=['Value', 'Year'])
    df['Year'] = df['Year'].astype(int)
    df['Decade'] = (df['Year'] // 10) * 10
    return df

df = load_data()

# --- SIDEBAR ---
st.sidebar.header("Settings")
all_years = sorted(df['Year'].unique())
selected_year = st.sidebar.select_slider("Waffle Year", options=all_years, value=2010)

all_decades = sorted(df['Decade'].unique())
selected_decades = st.sidebar.slider("Rose Decade Range", 
                                     min_value=int(min(all_decades)), 
                                     max_value=int(max(all_decades)), 
                                     value=(1950, 2010), step=10)

# --- DEBUG INFO (Check this if the chart is empty!) ---
st.sidebar.divider()
st.sidebar.write("### Data Debugger")
rural_count = len(df[df['Area'] == 'Rural'])
urban_count = len(df[df['Area'] == 'Urban'])
st.sidebar.write(f"Rural rows found: {rural_count}")
st.sidebar.write(f"Urban rows found: {urban_count}")

# --- MAIN LAYOUT ---
col1, col2 = st.columns(2)

with col1:
    st.header(f"Sex Ratio: {selected_year}")
    # Waffle logic
    w_data = df[(df['Year'] == selected_year) & (df['Area'] == 'Total')]
    m = w_data[w_data['Sex'] == 'Male']['Value'].sum()
    f = w_data[w_data['Sex'] == 'Female']['Value'].sum()
    
    if (m + f) > 0:
        male_pct = int((m / (m + f)) * 100)
        fig_w, ax_w = plt.subplots()
        x, y = np.meshgrid(np.arange(10), np.arange(10))
        colors = ['#3498db' if i < male_pct else '#ff69b4' for i in range(100)]
        ax_w.scatter(x.flatten(), y.flatten(), c=colors, marker='s', s=350, edgecolors='white')
        ax_w.axis('off')
        st.pyplot(fig_w)
        st.write(f"**Male:** {male_pct}% | **Female:** {100-male_pct}%")

with col2:
    st.header("Urban vs. Rural Gap")
    
    # 1. Filter by the range from the slider
    min_d, max_d = selected_decades
    rose_df = df[(df['Decade'] >= min_d) & (df['Decade'] <= max_d)]
    
    # 2. Filter for Area AND ensure we aren't double counting (Use 'Both Sexes')
    gap_data = rose_df[(rose_df['Area'].isin(['Urban', 'Rural'])) & (rose_df['Sex'] == 'Both Sexes')]
    
    if gap_data.empty:
        st.warning("No Urban/Rural data found for this range. The UN dataset often only has this for specific years (Census years).")
    else:
        # Group by Decade and Area
        pivot = gap_data.groupby(['Decade', 'Area'])['Value'].sum().unstack().fillna(0)
        
        # We define a fixed set of decades so the wedges don't "stretch"
        available_decades = [1950, 1960, 1970, 1980, 1990, 2000, 2010]
        # Only use decades that exist in our filtered pivot
        plot_decades = [d for d in available_decades if d in pivot.index]
        
        labels = [f"{d}s" for d in plot_decades]
        angles = np.linspace(0, 2 * np.pi, len(plot_decades), endpoint=False)
        width = (2 * np.pi / len(plot_decades)) * 0.7

        fig_r, ax_r = plt.subplots(subplot_kw={'projection': 'polar'})
        
        # Plot Rural (Gray)
        ax_r.bar(angles, pivot.loc[plot_decades, 'Rural'], width=width, color='lightgray', 
                 alpha=0.5, label='Rural', edgecolor='black')
        
        # Plot Urban (Blue)
        ax_r.bar(angles, pivot.loc[plot_decades, 'Urban'], width=width, color='#3498db', 
                 alpha=0.7, label='Urban', edgecolor='black')
        
        ax_r.set_theta_zero_location('N')
        ax_r.set_theta_direction(-1)
        ax_r.set_xticks(angles)
        ax_r.set_xticklabels(labels)
        ax_r.set_yticklabels([])
        ax_r.legend(loc='lower right', bbox_to_anchor=(1.3, 0.1))
        
        st.pyplot(fig_r)
