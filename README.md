# SafeNet 🔐

SafeNet is a Big Data-powered crime analysis and prediction system for Dehradun.  
It uses PySpark for scalable data processing, ML models for crime forecasting, and Streamlit for interactive dashboards.

## Features
- PySpark-based preprocessing and cleaning
- Crime trend visualizations with Pandas, Matplotlib, Folium
- ML models to predict police response time and resource allocation
- Interactive Streamlit dashboard with filters, bar charts, and heatmaps

## Technologies
- PySpark
- Pandas, Scikit-learn
- Streamlit, Folium, Seaborn
- HDFS, Hive

## Getting Started
Clone the repo and activate the `safenet_env` Conda environment:

```bash
git clone https://github.com/your-username/SafeNet.git
cd SafeNet
conda activate safenet_env
streamlit run src/crime_dashboard.py
