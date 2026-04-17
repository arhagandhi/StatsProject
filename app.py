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
    
    # Read the CSV - we use low_memory=False to handle mixed types in UN data
    df = pd.read_csv(file_name, low_memory=False)
    
    # Clean column names (strip spaces)
    df.columns = [str(c).strip() for c in df.columns]

    # --- THE DATA SANITIZER ---
    # 1. Identify Year, Sex, and Area columns dynamically
    def find_col(keywords):
        for col in df.columns:
            if any(key.lower() in str(col).lower() for key in keywords):
                return col
        return None

    year_col = find_col(['Year'])
    sex_col = find_col(['Sex'])
    area_col = find_col(['Area', 'Residence'])
    val_col = find_col(['Value', 'Number', 'Deaths'])
    country_col = find_col(['Country', 'Area'])

    # 2. Safety Check
    if not all([year_col, sex_col, area_col, val_col]):
        st.error(f"Could not map columns. Found: Year({year_col}), Sex({sex_col}), Area({area_col}), Value({val_col})")
        st.stop()

    # 3. Rename for internal code logic
    df = df.rename(columns={year_col: 'Year', sex_col: 'Sex', area_col: 'Area', val_col: 'Value', country_col: 'Country'})
    
    # 4. Filter for US and clean data types
    df = df[df['Country'].astype(str).str.contains('United States', na=False)]
    df['Value'] = pd.to_numeric(df['Value'], errors='coerce')
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
    df = df.dropna(subset=['Value', 'Year'])
    
    # 5. Clean text values
    df['Sex'] = df['Sex'].astype(str).str.strip()
    df['Area'] = df['Area'].astype(str).str.strip()
    df['Year'] = df['Year'].astype(int)
    df['Decade'] = (df['Year'] // 10) * 10
    
    return df

df = load_data()

# --- SIDEBAR ---
st.sidebar.header("Navigation")
all_years = sorted(df['Year'].unique())
selected_year = st.sidebar.select_slider("Select Year (Sex Ratio)", options=all_years, value=all_years[-1])

all_decades = sorted(df['Decade'].unique())
selected_decades = st.sidebar.select_slider("Select Decade Range (Urban/Rural)", 
                                           options=all_decades, 
                                           value=(min(all_decades), max(all_decades)))

# --- VISUALIZATIONS ---
col1, col2 = st.columns(2)

with col1:
    st.header(f"Sex Ratio: {selected_year}")
    # Waffle logic: use Total area for the sex ratio check
    w_data = df[(df['Year'] == selected_year) & (df['Area'].str.contains('Total', case=False, na=False))]
    m = w_data[w_data['Sex'].str.contains('Male', case=False, na=False)]['Value'].sum()
    f = w_data[w_data['Sex'].str.contains('Female', case=False, na=False)]['Value'].sum()
    
    if (m + f) > 0:
        male_pct = int((m / (m + f)) * 100)
        fig_w, ax_w = plt.subplots(figsize=(5,5))
        x, y = np.meshgrid(np.arange(10), np.arange(10))
        colors = ['#3498db' if i < male_pct else '#ff69b4' for i in range(100)]
        ax_w.scatter(x.flatten(), y.flatten(), c=colors, marker='s', s=400, edgecolors='white')
        ax_w.axis('off')
        st.pyplot(fig_w)
        st.markdown(f"**Male: {male_pct}%** | **Female: {100-male_pct}%**")
    else:
        st.warning("No disaggregated sex data for 'Total' area in this year.")

with col2:
    st.header("Urban vs. Rural Mortality")
    
    start_d, end_d = selected_decades
    rose_df = df[(df['Decade'] >= start_d) & (df['Decade'] <= end_d)]
    
    # Filter for Urban/Rural and use 'Both Sexes' for volume
    gap_data = rose_df[rose_df['Area'].isin(['Urban', 'Rural']) & rose_df['Sex'].str.contains('Both', na=False)]
    
    if gap_data.empty:
        st.warning("No Urban/Rural data found for this range.")
    else:
        pivot = gap_data.groupby(['Decade', 'Area'])['Value'].sum().unstack().fillna(0)
        labels = [f"{int(d)}s" for d in pivot.index]
        angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
        width = (2 * np.pi / len(labels)) * 0.7

        fig_r, ax_r = plt.subplots(subplot_kw={'projection': 'polar'})
        if 'Rural' in pivot.columns:
            ax_r.bar(angles, pivot['Rural'], width=width, color='lightgray', alpha=0.5, label='Rural')
        if 'Urban' in pivot.columns:
            ax_r.bar(angles, pivot['Urban'], width=width, color='#3498db', alpha=0.7, label='Urban')
        
        ax_r.set_theta_zero_location('N')
        ax_r.set_theta_direction(-1)
        ax_r.set_xticks(angles)
        ax_r.set_xticklabels(labels)
        ax_r.set_yticklabels([])
        ax_r.legend(loc='lower right', bbox_to_anchor=(1.3, 0.1))
        st.pyplot(fig_r)
