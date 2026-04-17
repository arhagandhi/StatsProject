import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

st.set_page_config(layout="wide", page_title="US Infant Mortality Study")

@st.cache_data
def load_data():
    # This looks for the file in the same folder on GitHub
    file_name = "UNdata_Export_20260417_223711907.csv"
    
    if not os.path.exists(file_name):
        st.error(f"File {file_name} not found in the repository!")
        st.stop()
        
    df = pd.read_csv(file_name)
    df = df[df['Country or Area'] == 'United States of America']
    df['Value'] = pd.to_numeric(df['Value'], errors='coerce')
    df = df.dropna(subset=['Value', 'Year'])
    df['Year'] = df['Year'].astype(int)
    df['Decade'] = (df['Year'] // 10) * 10
    return df

df = load_data()

# --- HEADER ---
st.title("📊 Statistics Project: US Infant Mortality (1948-Present)")
st.markdown("Exploring biological sex ratios and geographic healthcare gaps.")

# --- SIDEBAR ---
years = sorted(df['Year'].unique())
selected_year = st.sidebar.select_slider("Select Year", options=years, value=years[0])

# --- LAYOUT ---
col1, col2 = st.columns(2)

with col1:
    st.subheader(f"Sex Ratio ({selected_year})")
    year_data = df[(df['Year'] == selected_year) & (df['Area'] == 'Total')]
    m = year_data[year_data['Sex'] == 'Male']['Value'].sum()
    f = year_data[year_data['Sex'] == 'Female']['Value'].sum()
    
    if (m + f) > 0:
        male_pct = int((m / (m + f)) * 100)
        fig, ax = plt.subplots()
        x, y = np.meshgrid(np.arange(10), np.arange(10))
        colors = ['#3498db' if i < male_pct else '#e74c3c' for i in range(100)]
        ax.scatter(x.flatten(), y.flatten(), c=colors, marker='s', s=300)
        ax.axis('off')
        st.pyplot(fig)
        st.metric("Male Mortality %", f"{male_pct}%")
    else:
        st.warning("No data for this year.")

with col2:
    st.subheader("Urban vs. Rural Mortality Gap")
    rose_df = df[(df['Sex'] == 'Both Sexes') & (df['Area'].isin(['Urban', 'Rural']))]
    pivot = rose_df.groupby(['Decade', 'Area'])['Value'].sum().unstack().dropna()
    
    labels = [f"{int(d)}s" for d in pivot.index]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
    
    fig_rose, ax_rose = plt.subplots(subplot_kw={'projection': 'polar'})
    ax_rose.bar(angles, pivot['Rural'], width=0.5, color='gray', alpha=0.3, label='Rural')
    ax_rose.bar(angles, pivot['Urban'], width=0.5, color='blue', alpha=0.6, label='Urban')
    ax_rose.set_xticks(angles)
    ax_rose.set_xticklabels(labels)
    ax_rose.legend()
    st.pyplot(fig_rose)
