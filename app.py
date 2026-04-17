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
        st.error("CSV File not found in GitHub repository!")
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
    df = df.dropna(subset=['Value', 'Year'])
    
    df['Sex'] = df['Sex'].astype(str).str.strip()
    df['Area'] = df['Area'].astype(str).str.strip()
    df['Year'] = df['Year'].astype(int)
    df['Decade'] = (df['Year'] // 10) * 10
    
    return df

df = load_data()

# --- 2. HEADER ---
st.title("📊 US Infant Mortality: The 70-Year Mortality Clock")

# --- 3. LAYOUT ---
col_w, col_r = st.columns([1, 1.2])

# --- Q1: WAFFLE (Retains slider for specific year inspection) ---
with col_w:
    st.header("Biological Sex Ratio")
    all_years = sorted(df['Year'].unique())
    selected_year = st.select_slider("Inspect Year", options=all_years, value=all_years[-1])
    
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
        st.pyplot(fig_w)
        st.markdown(f"**{selected_year} Ratio:** ♂️ {male_pct}% | ♀️ {100-male_pct}%")

# --- Q2: THE 70-YEAR CLOCK ---
with col_r:
    st.header("The Urban-Rural Mortality Clock")
    st.write("Each wedge is a decade. Blue = Urban, Gray = Rural.")

    # Prepare data for all available decades
    gap_data = df[df['Area'].isin(['Urban', 'Rural']) & df['Sex'].str.contains('Both', na=False)]
    pivot = gap_data.groupby(['Decade', 'Area'])['Value'].sum().unstack().fillna(0)
    
    if pivot.empty:
        st.warning("No Urban/Rural data found to build the clock.")
    else:
        decades = sorted(pivot.index)
        labels = [f"{int(d)}s" for d in decades]
        num_decades = len(decades)
        
        # Setup angles: one slice per decade
        angles = np.linspace(0, 2 * np.pi, num_decades, endpoint=False)
        width = (2 * np.pi / num_decades) * 0.4 # Thinner wedges to fit side-by-side
        
        fig_r, ax_r = plt.subplots(figsize=(8, 8), subplot_kw={'projection': 'polar'})
        
        # Plot Urban and Rural side-by-side within each slice
        # Shift Urban left and Rural right from the center of the slice
        ax_r.bar(angles - width/2, pivot['Urban'], width=width, color='#3498db', alpha=0.8, label='Urban', edgecolor='black')
        ax_r.bar(angles + width/2, pivot['Rural'], width=width, color='#bdc3c7', alpha=0.8, label='Rural', edgecolor='black')
        
        # Formatting the Clock
        ax_r.set_theta_zero_location('N') # 12 o'clock start
        ax_r.set_theta_direction(-1)     # Clockwise time progression
        ax_r.set_xticks(angles)
        ax_r.set_xticklabels(labels, weight='bold', fontsize=12)
        ax_r.set_yticklabels([]) # Cleaner look
        
        ax_r.legend(loc='lower right', bbox_to_anchor=(1.3, 0.1))
        
        st.pyplot(fig_r)
        
        st.info("""
        **How to read the Clock:** As you move clockwise from the 1950s, the total size of the 'petals' shrinks, showing medical progress. 
        Notice the difference in size between Urban (Blue) and Rural (Gray) bars in each decade.
        """)

st.divider()
st.caption("Data Source: UN Population Division | Project Analysis: BMED 2400")
