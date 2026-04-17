import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# --- 1. CONFIG & DATA LOADING ---
st.set_page_config(layout="wide", page_title="US Infant Mortality BMED Study")

@st.cache_data
def load_data():
    file_name = "UNdata_Export_20260417_223711907.csv"
    if not os.path.exists(file_name):
        st.error(f"File {file_name} not found! Check your GitHub repo.")
        st.stop()
    
    df = pd.read_csv(file_name, low_memory=False)
    df.columns = [str(c).strip() for c in df.columns]

    def find_col(keywords):
        for col in df.columns:
            if any(key.lower() in str(col).lower() for key in keywords):
                return col
        return None

    df = df.rename(columns={
        find_col(['Year']): 'Year', 
        find_col(['Sex']): 'Sex', 
        find_col(['Area', 'Residence']): 'Area', 
        find_col(['Value', 'Number']): 'Value', 
        find_col(['Country', 'Area']): 'Country'
    })
    
    df = df[df['Country'].astype(str).str.contains('United States', na=False)]
    df['Value'] = pd.to_numeric(df['Value'], errors='coerce')
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
    df = df.dropna(subset=['Value', 'Year'])
    
    df['Sex'] = df['Sex'].astype(str).str.strip()
    df['Area'] = df['Area'].astype(str).str.strip()
    df['Year'] = df['Year'].astype(int)
    df['Decade'] = (df['Year'] // 10) * 10
    
    return df

df = load_data()

# --- 2. SIDEBAR CONTROLS ---
st.sidebar.header("📊 Interactive Controls")

# Waffle Slider (Single Year)
all_years = sorted(df['Year'].unique())
selected_year = st.sidebar.select_slider("1. Waffle Chart Year", options=all_years, value=all_years[-1])

# Rose Slider (Single Decade)
all_decades = sorted(df['Decade'].unique())
selected_decade = st.sidebar.select_slider("2. Rose Chart Decade", options=all_decades, value=all_decades[-1])

# --- 3. LAYOUT ---
col_w, col_r = st.columns(2)

# --- Q1: WAFFLE CHART ---
with col_w:
    st.header(f"Sex Ratio in {selected_year}")
    w_data = df[(df['Year'] == selected_year) & (df['Area'].str.contains('Total', case=False, na=False))]
    m = w_data[w_data['Sex'].str.contains('Male', case=False, na=False)]['Value'].sum()
    f = w_data[w_data['Sex'].str.contains('Female', case=False, na=False)]['Value'].sum()
    
    if (m + f) > 0:
        male_pct = int((m / (m + f)) * 100)
        fig_w, ax_w = plt.subplots(figsize=(6, 6))
        x, y = np.meshgrid(np.arange(10), np.arange(10))
        colors = ['#3498db' if i < male_pct else '#ff69b4' for i in range(100)]
        ax_w.scatter(x.flatten(), y.flatten(), c=colors, marker='s', s=450, edgecolors='white', linewidth=0.5)
        ax_w.axis('off')
        ax_w.set_aspect('equal')
        st.pyplot(fig_w)
        st.write(f"**Male: {male_pct}% | Female: {100-male_pct}%**")
    else:
        st.warning("No data for this year.")

# --- Q2: SIDE-BY-SIDE ROSE CHART ---
with col_r:
    st.header(f"Urban vs. Rural Comparison ({selected_decade}s)")
    
    # Filter for the ONE selected decade
    rose_df = df[(df['Decade'] == selected_decade) & 
                 (df['Area'].isin(['Urban', 'Rural'])) & 
                 (df['Sex'].str.contains('Both', na=False))]
    
    if rose_df.empty:
        st.warning(f"No Urban/Rural data found for the {selected_decade}s.")
    else:
        # Get values
        u_val = rose_df[rose_df['Area'] == 'Urban']['Value'].sum()
        r_val = rose_df[rose_df['Area'] == 'Rural']['Value'].sum()
        
        # Plotting
        fig_r, ax_r = plt.subplots(figsize=(6, 6), subplot_kw={'projection': 'polar'})
        
        # We define two fixed angles for Urban and Rural
        angles = [np.deg2rad(45), np.deg2rad(135)] 
        vals = [u_val, r_val]
        colors = ['#3498db', '#bdc3c7']
        labels = ['Urban', 'Rural']
        
        bars = ax_r.bar(angles, vals, width=0.6, color=colors, edgecolor='black', alpha=0.8)
        
        # Formatting
        ax_r.set_theta_zero_location('N')
        ax_r.set_xticks(angles)
        ax_r.set_xticklabels(labels, weight='bold', fontsize=12)
        ax_r.set_yticklabels([]) # Hide radii
        
        # Add labels on top of the bars
        for bar, val in zip(bars, vals):
            height = bar.get_height()
            ax_r.text(bar.get_x() + bar.get_width()/2, height + (height*0.05), 
                      f'{int(val):,}', ha='center', va='bottom', fontsize=10, weight='bold')

        st.pyplot(fig_r)
        
        # Analysis Text
        gap = r_val - u_val
        if gap > 0:
            st.error(f"**Rural Penalty:** There were {int(gap):,} more deaths in Rural areas than Urban areas during this decade.")
        else:
            st.success(f"**Urban Gap:** Urban areas recorded {int(abs(gap)):,} more deaths than Rural areas.")

st.divider()
st.caption("BIOS 4505 / BMED 2400 | Data Source: UN Population Division")
        """)

st.divider()
st.caption("Data Source: UN Population Division | Analysis for BIOS 4505 / BMED 2400")
