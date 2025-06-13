# crime_dashboard.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
from sklearn.ensemble import RandomForestRegressor

# ── 1) LOAD DATA ────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    file_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'crime_data_dehradun.csv')
    return pd.read_csv(file_path)

df = load_data()

# ── 2) SIDEBAR FILTERS ──────────────────────────────────────────────────────────
st.title("SafeNet: Dehradun Crime Data Dashboard")
st.sidebar.header("Filters")
areas = df["area"].unique().tolist()
years = df["year"].unique().astype(int).tolist()

selected_areas = st.sidebar.multiselect("Area", areas, default=areas)
selected_years = st.sidebar.multiselect("Year", years, default=years)

mask = df["area"].isin(selected_areas) & df["year"].isin(selected_years)
filtered_df = df[mask]

# ── 3) SECTION 1: CRIME TYPE DISTRIBUTION ──────────────────────────────────────
st.subheader("Crime Type Distribution")
ct_counts = filtered_df["crime_type"].value_counts()
fig1, ax1 = plt.subplots()
sns.barplot(x=ct_counts.values, y=ct_counts.index, palette="Reds_r", ax=ax1)
ax1.set_xlabel("Count")
ax1.set_ylabel("Crime Type")
st.pyplot(fig1)

# ── 4) SECTION 2: AREA-WISE CRIME COUNT ────────────────────────────────────────
st.subheader("Area-wise Crime Count")
area_counts = filtered_df["area"].value_counts()
fig2, ax2 = plt.subplots()
sns.barplot(x=area_counts.values, y=area_counts.index, palette="Blues_r", ax=ax2)
ax2.set_xlabel("Count")
ax2.set_ylabel("Area")
st.pyplot(fig2)

# ── 5) SECTION 3: YEAR-WISE TREND ──────────────────────────────────────────────
st.subheader("Year-wise Crime Trend")
year_counts = filtered_df.groupby("year").size().sort_index()
fig3, ax3 = plt.subplots()
ax3.plot(year_counts.index, year_counts.values, marker='o', color='navy')
ax3.set_xlabel("Year")
ax3.set_ylabel("Crime Count")
ax3.grid(True)
st.pyplot(fig3)

# ── 6) SECTION 4: CRIME HEATMAP ─────────────────────────────────────────────────
st.subheader("Crime Density Heatmap")
center = [filtered_df["latitude"].mean(), filtered_df["longitude"].mean()]
m = folium.Map(location=center, zoom_start=12)
heat_data = filtered_df[["latitude", "longitude"]].dropna().values.tolist()
HeatMap(heat_data, radius=8, blur=12,
        gradient={0.4:'blue',0.65:'lime',1:'red'}).add_to(m)
st_folium(m, width=700, height=500)

# ── 7) PREPARE AREA STATS FOR MODELS ────────────────────────────────────────────
area_stats = (
    filtered_df
    .groupby("area")
    .agg(
        crime_count=("crime_id", "count"),
        avg_response_time=("response_time_min", "mean"),
        avg_cctv=("no_of_cctv", "mean")
    )
    .reset_index()
)

# ── 8) OFFICER REQUIREMENT ML MODEL ────────────────────────────────────────────
st.subheader("Predicted Police Officers Needed by Area")

# Label: crime_count / 10
area_stats["recommended_officers"] = (area_stats["crime_count"] / 10).astype(int)

X_off = area_stats[["crime_count", "avg_response_time", "avg_cctv"]]
y_off = area_stats["recommended_officers"]

rf_off = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
rf_off.fit(X_off, y_off)

area_stats["pred_officers"] = rf_off.predict(X_off).round().astype(int)

# Plot
fig4, ax4 = plt.subplots(figsize=(8,6))
df_off = area_stats.sort_values("pred_officers", ascending=True)
bars = ax4.barh(df_off["area"], df_off["pred_officers"], color="teal")
for bar in bars:
    w = bar.get_width()
    ax4.text(w-1, bar.get_y() + bar.get_height()/2, str(w),
             va='center', ha='right', color='white')
ax4.set_xlabel("Officers")
st.pyplot(fig4)

# ── 9) CCTV REQUIREMENT ML MODEL ────────────────────────────────────────────────
st.subheader("Predicted CCTVs Needed by Area")

# Heuristic: double current CCTV count
area_stats["needed_cctv"] = (area_stats["avg_cctv"] * 2).astype(int)

X_cctv = X_off.copy()
y_cctv = area_stats["needed_cctv"]

rf_cctv = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
rf_cctv.fit(X_cctv, y_cctv)

area_stats["pred_cctv"] = rf_cctv.predict(X_cctv).round().astype(int)

# Plot
fig5, ax5 = plt.subplots(figsize=(8,6))
df_cv = area_stats.sort_values("pred_cctv", ascending=True)
bars2 = ax5.barh(df_cv["area"], df_cv["pred_cctv"], color="purple")
for bar in bars2:
    w = bar.get_width()
    ax5.text(w-1, bar.get_y() + bar.get_height()/2, str(w),
             va='center', ha='right', color='white')
ax5.set_xlabel("CCTVs")
st.pyplot(fig5)
