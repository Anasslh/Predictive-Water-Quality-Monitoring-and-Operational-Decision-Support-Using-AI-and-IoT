import os
import sys
import json
import pyodbc
from dotenv import load_dotenv

load_dotenv()

DB_CONN_STR = os.getenv("DB_CONN_STR")
if not DB_CONN_STR:
    raise ValueError("Missing DB_CONN_STR. Please check your .env file.")

def run_etl_pipeline():
    # 1. Dynamically capture the file path from the terminal command
    if len(sys.argv) > 1:
        # If the user provided a file path, use it
        target_file = sys.argv[1]
    else:
        # If they forgot, default to the local test file
        target_file = os.path.join("exports", "EC.jsonl")
        print("No file path provided. Defaulting to exports/EC.jsonl")

    # Resolve the absolute path to make it bulletproof regardless of terminal location
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(BASE_DIR, target_file)
    
    print(f"Extracting data from: {file_path} ...")
    
    records_to_insert = []
    
    try:
        with open(file_path, 'r') as file:
            for line in file:
                if not line.strip():
                    continue
                    
                record = json.loads(line)
                shap_json_str = json.dumps(record.get('shap_top_features', []))
                
                row = (
                    record.get('timestamp'),
                    record.get('parameter_name'),
                    record.get('predicted_value'),
                    record.get('actual_value'),
                    shap_json_str,
                    record.get('is_anomaly'),
                    record.get('anomaly_score'),
                    record.get('retrain_alert')
                )
                records_to_insert.append(row)
    except FileNotFoundError:
        print(f"Error: Could not find the file at {file_path}. Make sure it exists.")
        return

    if not records_to_insert:
        print("No new records found in the JSONL file.")
        return

    print(f"Transform complete. Loading {len(records_to_insert)} records into SQL Server...")
    
    insert_query = """
        INSERT INTO Fact_WaterQuality 
        (MeasurementTimestamp, ParameterName, PredictedValue, ActualValue, 
         ShapTopFeatures, IsAnomaly, AnomalyScore, RetrainAlert)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """
    
    try:
        conn = pyodbc.connect(DB_CONN_STR)
        cursor = conn.cursor()
        cursor.executemany(insert_query, records_to_insert)
        conn.commit()
        print("ETL Pipeline completed successfully. Data is now in the Warm tier.")
    except Exception as e:
        print(f"Database error occurred: {e}")
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    run_etl_pipeline()