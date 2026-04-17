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
        st.error(f"File {file_name} not found! Ensure it's in your GitHub repo.")
        st.stop()
    
    df = pd.read_csv(file_name, low_memory=False)
    df.columns = [str(c).strip() for c in df.columns]

    # Dynamic Column Detection
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

    # Standardizing Names
    df = df.rename(columns={
        year_col: 'Year', 
        sex_col: 'Sex', 
        area_col: 'Area', 
        val_col: 'Value', 
        country_col: 'Country'
    })
    
    # Cleaning
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

# --- 2. HEADER & STYLE ---
st.title("📊 US Infant Mortality: Biological vs. Geographic Trends")
st.markdown("""
<style>
.big-font { font-size:22px !important; font-weight:bold; color: #333;}
</style>
""", unsafe_allow_html=True)

# --- 3. SIDEBAR NAVIGATION ---
st.sidebar.header("Navigation")
all_years = sorted(df['Year'].unique())
selected_year = st.sidebar.select_slider("Select Year (Sex Ratio)", options=all_years, value=all_years[-1])

all_decades = sorted(df['Decade'].unique())
selected_decades = st.sidebar.select_slider("Select Decade Range (Gap Analysis)", 
                                           options=all_decades, 
                                           value=(min(all_decades), max(all_decades)))

st.sidebar.divider()
st.sidebar.info("**Note:** The Urban/Rural gap is often most visible in decades like the 1970s and 1980s due to UN reporting cycles.")

# --- 4. MAIN LAYOUT ---
col_w, col_r = st.columns(2)

# --- Q1: THE BABY WAFFLE ---
with col_w:
    st.header(f"Q1: The Biological Sex Ratio ({selected_year})")
    
    # Filter for year and 'Total' area
    w_data = df[(df['Year'] == selected_year) & (df['Area'].str.contains('Total', case=False, na=False))]
    m = w_data[w_data['Sex'].str.contains('Male', case=False, na=False)]['Value'].sum()
    f = w_data[w_data['Sex'].str.contains('Female', case=False, na=False)]['Value'].sum()
    
    if (m + f) > 0:
        male_pct = int((m / (m + f)) * 100)
        
        fig_w, ax_w = plt.subplots(figsize=(6, 6))
        x, y = np.meshgrid(np.arange(10), np.arange(10))
        # Blue for Male, Pink for Female
        colors = ['#3498db' if i < male_pct else '#ff69b4' for i in range(100)]
        
        ax_w.scatter(x.flatten(), y.flatten(), c=colors, marker='s', s=450, edgecolors='white', linewidth=0.5)
        ax_w.axis('off')
        ax_w.set_aspect('equal')
        ax_w.text(4.5, 10, "🍼 Biological Vulnerability", ha='center', fontsize=14, weight='bold')
        
        st.pyplot(fig_w)
        st.markdown(f'<p class="big-font">♂️ Male: <span style="color:#3498db">{male_pct}%</span> | ♀️ Female: <span style="color:#ff69b4">{100-male_pct}%</span></p>', unsafe_allow_html=True)
    else:
        st.warning(f"No sex-specific data for {selected_year} in 'Total' area.")

# --- Q2: THE OVERLAPPING ROSE ---
with col_r:
    st.header("Q2: The Urban-Rural 'Penalty'")
    
    start_d, end_d = selected_decades
    rose_df = df[(df['Decade'] >= start_d) & (df['Decade'] <= end_d)]
    
    # Use 'Both Sexes' for total volume check
    gap_data = rose_df[rose_df['Area'].isin(['Urban', 'Rural']) & rose_df['Sex'].str.contains('Both', na=False)]
    
    if gap_data.empty:
        st.warning("No Urban/Rural data found for this range. Try including the 1970s.")
    else:
        pivot = gap_data.groupby(['Decade', 'Area'])['Value'].sum().unstack().fillna(0)
        labels = [f"{int(d)}s" for d in pivot.index]
        angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
        width = (2 * np.pi / len(labels)) * 0.7

        fig_r, ax_r = plt.subplots(figsize=(6, 6), subplot_kw={'projection': 'polar'})
        
        # Plot RURAL first (Background - Gray)
        if 'Rural' in pivot.columns:
            ax_r.bar(angles, pivot['Rural'], width=width, 
                     color='#bdc3c7', alpha=0.4, label='Rural Mortality', 
                     edgecolor='#2c3e50', linewidth=1.2)
        
        # Plot URBAN second (Foreground - Blue)
        if 'Urban' in pivot.columns:
            ax_r.bar(angles, pivot['Urban'], width=width, 
                     color='#3498db', alpha=0.7, label='Urban Mortality', 
                     edgecolor='#2980b9', linewidth=1.2)
        
        ax_r.set_theta_zero_location('N')
        ax_r.set_theta_direction(-1)
        ax_r.set_xticks(angles)
        ax_r.set_xticklabels(labels, weight='bold')
        ax_r.set_yticklabels([]) # Cleaner look
        ax_r.legend(loc='lower right', bbox_to_anchor=(1.3, 0.1))
        
        st.pyplot(fig_r)
        st.markdown("""
        **Gap Analysis:** The gray area extending beyond the blue represents the 'Rural Penalty'. 
        If the blue core shrinks faster than the gray shell, medical advancement is 
        concentrated in urban centers.
        """)

st.divider()
st.caption("Data Source: UN Population Division | Analysis for BIOS 4505 / BMED 2400")
