import os
import pyodbc
import pandas as pd
import json
import streamlit as st
from dotenv import load_dotenv

# Ensure environment variables are loaded
load_dotenv()

@st.cache_data(ttl=3600)  # Cache the query results for 1 hour to keep the dashboard fast
def fetch_historical_parameter_data(parameter_name):
    """
    Connects to the local SQL Server Warm tier and pulls the 12-month historical 
    data for a specific parameter to render in the Streamlit dashboard.
    """
    db_conn_str = os.getenv("DB_CONN_STR")
    if not db_conn_str:
        st.error("Database connection string is missing. Check the .env file.")
        return pd.DataFrame()

    query = """
    SELECT 
        CAST(MeasurementTimestamp AS VARCHAR(50)) AS timestamp,
        PredictedValue AS predicted_value,
        ActualValue AS actual_value,
        IsAnomaly AS is_anomaly,
        AnomalyScore AS anomaly_score,
        ShapTopFeatures AS shap_top_features,
        RetrainAlert AS retrain_alert
    FROM Fact_WaterQuality
    WHERE ParameterName = ?
    ORDER BY MeasurementTimestamp ASC
"""

    try:
        conn = pyodbc.connect(db_conn_str)
        df = pd.read_sql(query, conn, params=[parameter_name])
        conn.close()

        if df.empty:
            return df

        # Ensure timestamps are strictly typed as datetime objects
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Convert the SQL BIT (0/1) back to boolean (False/True)
        df['is_anomaly'] = df['is_anomaly'].astype(bool)

        # Parse the JSON strings back into native Python lists/dictionaries
        df['shap_top_features'] = df['shap_top_features'].apply(
            lambda x: json.loads(x) if pd.notnull(x) else []
        )

        return df

    except Exception as e:
        st.error(f"Warm Tier Database Error: {e}")
        return pd.DataFrame()