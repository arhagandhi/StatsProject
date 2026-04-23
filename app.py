import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

st.set_page_config(layout="wide", page_title="US Infant Mortality Study")

@st.cache_data
def load_data():
    # Target the new filename
    file_name = "United nations rural vs urban.csv"
    
    if not os.path.exists(file_name):
        st.error(f"File '{file_name}' not found! Make sure you uploaded it to GitHub with this exact name.")
        st.stop()
    
    # Load data
    df = pd.read_csv(file_name, low_memory=False)
    
    # Standardize headers
    df.columns = [str(c).strip() for c in df.columns]

    # Map the columns
    # This matches the structure of your new CSV: ['Year', 'Area', 'Sex', 'Value']
    df = df.rename(columns={
        'Country or Area': 'Country',
        'Year': 'Year',
        'Area': 'Area',
        'Sex': 'Sex',
        'Value': 'Value'
    })

    # Clean data
    df = df[df['Country'].astype(str).str.contains('United States', case=False, na=False)]
    df['Value'] = pd.to_numeric(df['Value'], errors='coerce')
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
    df = df.dropna(subset=['Value', 'Year'])
    
    # Standardize text for filtering
    df['Sex'] = df['Sex'].astype(str).str.strip()
    df['Area'] = df['Area'].astype(str).str.strip()
    df['Decade'] = (df['Year'].astype(int) // 10) * 10
    
    return df

df = load_data()

st.title("📊 US Infant Mortality: The 70-Year Mortality Clock")
st.markdown("Using official United Nations Rural vs. Urban statistics.")

# --- 1. LAYOUT ---
col_w, col_r = st.columns([1, 1.2])

# --- Q1: WAFFLE CHART (Sex Ratio) ---
with col_w:
    st.header("Biological Sex Ratio")
    years = sorted(df['Year'].unique().astype(int))
    sel_year = st.select_slider("Select Year for Waffle Chart", options=years, value=years[0])
    
    # Filter for the year. We sum Male and Female regardless of Area to get the national ratio.
    w_data = df[df['Year'] == sel_year]
    
    m = w_data[w_data['Sex'].str.fullmatch('Male', case=False)]['Value'].sum()
    f = w_data[w_data['Sex'].str.fullmatch('Female', case=False)]['Value'].sum()

    if (m + f) > 0:
        male_pct = int((m / (m + f)) * 100)
        fig_w, ax_w = plt.subplots(figsize=(6, 6))
        x, y = np.meshgrid(np.arange(10), np.arange(10))
        # Blue for Male, Pink for Female
        colors = ['#3498db' if i < male_pct else '#ff69b4' for i in range(100)]
        ax_w.scatter(x.flatten(), y.flatten(), c=colors, marker='s', s=450, edgecolors='white', linewidth=0.5)
        ax_w.axis('off')
        ax_w.set_aspect('equal')
        st.pyplot(fig_w)
        st.markdown(f"**{sel_year} Ratio:** ♂️ Male {male_pct}% | ♀️ Female {100-male_pct}%")
    else:
        st.warning(f"No Male/Female data found for {sel_year}.")

# --- Q2: THE 70-YEAR CLOCK (Urban vs Rural) ---
with col_r:
    st.header("Urban-Rural Mortality Clock")
    st.write("Each wedge represents a decade. Time progresses clockwise.")
    
    # Filter for Urban/Rural and 'Both Sexes' to get total deaths per area
    r_data = df[df['Area'].isin(['Urban', 'Rural']) & df['Sex'].str.contains('Both', case=False)]
    
    if r_data.empty:
        st.warning("No Urban/Rural data found in this file. Check if 'Area' column contains 'Urban' or 'Rural'.")
    else:
        pivot = r_data.groupby(['Decade', 'Area'])['Value'].sum().unstack().fillna(0)
        
        decades = sorted(pivot.index)
        labels = [f"{int(d)}s" for d in decades]
        angles = np.linspace(0, 2 * np.pi, len(decades), endpoint=False)
        width = (2 * np.pi / len(decades)) * 0.35
        
        fig_r, ax_r = plt.subplots(figsize=(8, 8), subplot_kw={'projection': 'polar'})
        
        # Plot Urban and Rural side-by-side
        if 'Urban' in pivot.columns:
            ax_r.bar(angles - width/2, pivot['Urban'], width=width, color='#3498db', label='Urban', edgecolor='black', alpha=0.8)
        if 'Rural' in pivot.columns:
            ax_r.bar(angles + width/2, pivot['Rural'], width=width, color='#bdc3c7', label='Rural', edgecolor='black', alpha=0.8)
        
        ax_r.set_theta_zero_location('N')
        ax_r.set_theta_direction(-1)
        ax_r.set_xticks(angles)
        ax_r.set_xticklabels(labels, weight='bold', fontsize=12)
        ax_r.set_yticklabels([])
        ax_r.legend(loc='lower right', bbox_to_anchor=(1.3, 0.1))
        
        st.pyplot(fig_r)

# --- DEBUG EXPANDER ---
with st.expander("Show Data Preview"):
    st.write("Rows in file:", len(df))
    st.write("Areas found:", df['Area'].unique())
    st.write("Sexes found:", df['Sex'].unique())
    st.dataframe(df.head(20))
