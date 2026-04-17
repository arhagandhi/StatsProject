import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

st.set_page_config(layout="wide", page_title="Infant Mortality Study")

@st.cache_data
def load_data():
    file_name = "UNdata_Export_20260417_223711907.csv"
    if not os.path.exists(file_name):
        st.error(f"File '{file_name}' not found in GitHub!")
        st.stop()
    
    # Load data and strip any weird spaces from headers
    df = pd.read_csv(file_name, low_memory=False)
    df.columns = [str(c).strip() for c in df.columns]

    # --- THE SMART SCANNER ---
    # We find the columns by searching for keywords inside the data rows
    def find_column_by_content(keywords):
        for col in df.columns:
            sample = df[col].astype(str).str.lower()
            if any(sample.str.contains(key.lower()).any() for key in keywords):
                return col
        return None

    col_sex = find_column_by_content(['male', 'female', 'both sexes'])
    col_area = find_column_by_content(['urban', 'rural', 'total'])
    col_country = find_column_by_content(['united states'])
    col_year = next((c for c in df.columns if 'year' in c.lower()), None)
    col_val = next((c for c in df.columns if 'value' in c.lower() or 'number' in c.lower()), df.columns[-1])

    # Rename for internal logic
    df = df.rename(columns={
        col_sex: 'Sex', col_area: 'Area', 
        col_year: 'Year', col_val: 'Value', col_country: 'Country'
    })

    # Filter for USA and clean numbers
    df = df[df['Country'].astype(str).str.contains('United States', case=False, na=False)]
    df['Value'] = pd.to_numeric(df['Value'], errors='coerce')
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
    df = df.dropna(subset=['Value', 'Year'])
    
    # Standardize the text
    df['Sex'] = df['Sex'].astype(str).str.strip()
    df['Area'] = df['Area'].astype(str).str.strip()
    df['Decade'] = (df['Year'].astype(int) // 10) * 10
    
    return df

try:
    df = load_data()
except Exception as e:
    st.error(f"Load Error: {e}")
    st.stop()

st.title("👶 US Infant Mortality Analysis")

# --- 1. WAFFLE CHART (Left) ---
col_w, col_r = st.columns([1, 1.2])

with col_w:
    st.header("Sex Ratio Waffle")
    years = sorted(df['Year'].unique().astype(int))
    sel_year = st.select_slider("Select Year", options=years, value=years[-1])
    
    # Filter: We need Male/Female for the selected year
    # We try to find the 'Total' area rows first
    w_data = df[(df['Year'] == sel_year) & (df['Area'].str.contains('total', case=False, na=False))]
    
    # If 'Total' area isn't found, we sum Urban + Rural
    if w_data.empty:
        w_data = df[df['Year'] == sel_year]

    m = w_data[w_data['Sex'].str.contains('male', case=False, na=False) & 
               ~w_data['Sex'].str.contains('both', case=False, na=False)]['Value'].sum()
    f = w_data[w_data['Sex'].str.contains('female', case=False, na=False)]['Value'].sum()

    if (m + f) > 0:
        male_pct = int((m / (m + f)) * 100)
        fig_w, ax_w = plt.subplots()
        x, y = np.meshgrid(np.arange(10), np.arange(10))
        colors = ['#3498db' if i < male_pct else '#ff69b4' for i in range(100)]
        ax_w.scatter(x.flatten(), y.flatten(), c=colors, marker='s', s=350, edgecolors='white')
        ax_w.axis('off')
        st.pyplot(fig_w)
        st.write(f"**{sel_year} Ratio:** ♂️ {male_pct}% | ♀️ {100-male_pct}%")
    else:
        st.warning("Could not calculate Sex Ratio for this year.")

# --- 2. THE FULL CLOCK (Right) ---
with col_r:
    st.header("70-Year Mortality Clock")
    
    # Filter for Urban/Rural and 'Both Sexes'
    r_data = df[df['Area'].str.contains('urban|rural', case=False, na=False) & 
                df['Sex'].str.contains('both', case=False, na=False)]
    
    pivot = r_data.groupby(['Decade', 'Area'])['Value'].sum().unstack().fillna(0)
    
    # Identify which columns are Urban vs Rural
    urban_col = next((c for c in pivot.columns if 'urban' in c.lower()), None)
    rural_col = next((c for c in pivot.columns if 'rural' in c.lower()), None)

    if urban_col and rural_col:
        decades = sorted(pivot.index)
        angles = np.linspace(0, 2 * np.pi, len(decades), endpoint=False)
        width = (2 * np.pi / len(decades)) * 0.35
        
        fig_r, ax_r = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(8,8))
        
        # Plot Side-by-Side
        ax_r.bar(angles - width/2, pivot[urban_col], width=width, color='#3498db', label='Urban', edgecolor='black')
        ax_r.bar(angles + width/2, pivot[rural_col], width=width, color='#bdc3c7', label='Rural', edgecolor='black')
        
        ax_r.set_theta_zero_location('N')
        ax_r.set_theta_direction(-1)
        ax_r.set_xticks(angles)
        ax_r.set_xticklabels([f"{int(d)}s" for d in decades], weight='bold')
        ax_r.set_yticklabels([])
        ax_r.legend(loc='lower right', bbox_to_anchor=(1.3, 0.1))
        st.pyplot(fig_r)
    else:
        st.warning("Urban/Rural columns not identified in dataset.")

# --- 3. THE EMERGENCY DIAGNOSTIC ---
with st.expander("DEBUG: See what the code found in your CSV"):
    st.write("Unique Areas found:", df['Area'].unique())
    st.write("Unique Sexes found:", df['Sex'].unique())
    st.dataframe(df.head(20))
