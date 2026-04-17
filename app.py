import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

st.set_page_config(layout="wide", page_title="US Infant Mortality Study")

@st.cache_data
def load_data():
    file_name = "UNdata_Export_20260417_223711907.csv"
    if not os.path.exists(file_name):
        st.error(f"File {file_name} not found!")
        st.stop()
    
    # Read and skip the first few rows if UN metadata is present
    df = pd.read_csv(file_name)
    
    # 1. Standardize column names to lowercase for easier filtering
    df.columns = [c.strip() for c in df.columns]
    
    # 2. Filter for USA only
    df = df[df['Country or Area'] == 'United States of America']
    
    # 3. CRITICAL: Convert 'Value' to numeric and drop NaN
    df['Value'] = pd.to_numeric(df['Value'], errors='coerce')
    df = df.dropna(subset=['Value', 'Year'])
    
    # 4. Clean text columns
    df['Area'] = df['Area'].str.strip()
    df['Sex'] = df['Sex'].str.strip()
    
    df['Year'] = df['Year'].astype(int)
    df['Decade'] = (df['Year'] // 10) * 10
    return df

df = load_data()

# --- SIDEBAR ---
st.sidebar.header("Interactive Controls")
all_years = sorted(df['Year'].unique())
selected_year = st.sidebar.select_slider("Waffle Year", options=all_years, value=2010)

all_decades = sorted(df['Decade'].unique())
start_d, end_d = st.sidebar.select_slider("Rose Decade Range", options=all_decades, value=(1950, 2010))

# Debugging Table in Sidebar
st.sidebar.divider()
st.sidebar.write("### Data Snapshot")
st.sidebar.write(df[['Year', 'Area', 'Sex', 'Value']].head(10))

# --- MAIN LAYOUT ---
col1, col2 = st.columns(2)

with col1:
    st.header(f"Sex Ratio: {selected_year}")
    # Waffle logic - Blue/Pink
    w_data = df[(df['Year'] == selected_year) & (df['Area'] == 'Total')]
    m = w_data[w_data['Sex'] == 'Male']['Value'].sum()
    f = w_data[w_data['Sex'] == 'Female']['Value'].sum()
    
    if (m + f) > 0:
        male_pct = int((m / (m + f)) * 100)
        fig_w, ax_w = plt.subplots(figsize=(5,5))
        x, y = np.meshgrid(np.arange(10), np.arange(10))
        colors = ['#3498db' if i < male_pct else '#ff69b4' for i in range(100)]
        ax_w.scatter(x.flatten(), y.flatten(), c=colors, marker='s', s=400, edgecolors='white')
        ax_w.axis('off')
        st.pyplot(fig_w)
        st.write(f"**Male:** {male_pct}% | **Female:** {100-male_pct}%")

with col2:
    st.header("Urban vs. Rural Gap")
    
    # Filter for selected decades and strictly Urban/Rural areas
    rose_df = df[(df['Decade'] >= start_d) & (df['Decade'] <= end_d)]
    # Use 'Both Sexes' to get the total volume per area
    gap_data = rose_df[(rose_df['Area'].isin(['Urban', 'Rural'])) & (rose_df['Sex'] == 'Both Sexes')]
    
    if gap_data.empty:
        st.warning("No Urban/Rural data found for this range. (Try 1960s-1970s).")
    else:
        # Pivot the data properly
        pivot = gap_data.groupby(['Decade', 'Area'])['Value'].sum().unstack().fillna(0)
        
        dec_labels = [f"{int(d)}s" for d in pivot.index]
        angles = np.linspace(0, 2 * np.pi, len(dec_labels), endpoint=False)
        width = (2 * np.pi / len(dec_labels)) * 0.7

        fig_r, ax_r = plt.subplots(figsize=(6,6), subplot_kw={'projection': 'polar'})
        
        # Plot Rural (Background) then Urban (Foreground)
        ax_r.bar(angles, pivot['Rural'], width=width, color='lightgray', alpha=0.5, label='Rural', edgecolor='black')
        ax_r.bar(angles, pivot['Urban'], width=width, color='#3498db', alpha=0.7, label='Urban', edgecolor='black')
        
        ax_r.set_theta_zero_location('N')
        ax_r.set_theta_direction(-1)
        ax_r.set_xticks(angles)
        ax_r.set_xticklabels(dec_labels)
        ax_r.set_yticklabels([])
        ax_r.legend(loc='lower right', bbox_to_anchor=(1.3, 0.1))
        
        st.pyplot(fig_r)
