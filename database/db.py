"""
Database Access Module for Intelligent Dementia Detection System
Provides lightweight, parameterized SQLite3 operations for storing and retrieving
patient clinical records and machine learning staging predictions.
"""

import os
import sqlite3
import pandas as pd
from typing import Optional, Dict, Any

# Default database location inside database/ directory
DEFAULT_DB_PATH = os.path.join(os.path.dirname(__file__), "patients.db")
SCHEMA_PATH = os.path.join(os.path.dirname(__file__), "schema.sql")


def get_connection(db_path: Optional[str] = None) -> sqlite3.Connection:
    """
    Establishes and returns an active SQLite database connection.
    Enforces foreign key constraints and configures sqlite3.Row row factory.
    
    Args:
        db_path: Path to the SQLite database file (defaults to database/patients.db)
        
    Returns:
        sqlite3.Connection object
    """
    target_path = db_path if db_path is not None else DEFAULT_DB_PATH
    
    # Ensure parent directory exists for file-based databases
    if target_path != ":memory:":
        os.makedirs(os.path.dirname(os.path.abspath(target_path)), exist_ok=True)
        
    conn = sqlite3.connect(target_path)
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: Optional[str] = None) -> None:
    """
    Initializes the database schema by executing schema.sql.
    Safe to run repeatedly (idempotent).
    
    Args:
        db_path: Optional database path override (e.g. for testing)
    """
    conn = get_connection(db_path)
    try:
        if os.path.exists(SCHEMA_PATH):
            with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
                schema_sql = f.read()
        else:
            # Fallback embedded schema if schema.sql is not found
            schema_sql = """
            PRAGMA foreign_keys = ON;
            CREATE TABLE IF NOT EXISTS patients (
                patient_id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                patient_name TEXT DEFAULT 'Anonymous',
                gender TEXT NOT NULL CHECK (gender IN ('Female', 'Male')),
                age REAL NOT NULL CHECK (age >= 18.0 AND age <= 105.0),
                educ_level INTEGER NOT NULL CHECK (educ_level >= 1 AND educ_level <= 5),
                ses REAL NOT NULL CHECK (ses >= 1.0 AND ses <= 5.0),
                mmse REAL NOT NULL CHECK (mmse >= 0.0 AND mmse <= 30.0),
                etiv REAL NOT NULL CHECK (etiv >= 800.0 AND etiv <= 2500.0),
                nwbv REAL NOT NULL CHECK (nwbv >= 0.500 AND nwbv <= 0.950),
                asf REAL NOT NULL CHECK (asf >= 0.700 AND asf <= 1.800)
            );
            CREATE TABLE IF NOT EXISTS predictions (
                prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id INTEGER NOT NULL,
                predicted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                prediction_label TEXT NOT NULL,
                probability REAL NOT NULL CHECK (probability >= 0.0 AND probability <= 1.0),
                risk_level TEXT,
                model_name TEXT NOT NULL,
                FOREIGN KEY (patient_id) REFERENCES patients(patient_id) ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS idx_predictions_patient_id ON predictions(patient_id);
            """
        conn.executescript(schema_sql)
        conn.commit()
    finally:
        conn.close()


def add_patient(data: Dict[str, Any], db_path: Optional[str] = None) -> int:
    """
    Inserts a new patient record with demographic and clinical parameters.
    
    Args:
        data: Dictionary containing patient fields (patient_name, gender, age,
              educ_level, ses, mmse, etiv, nwbv, asf)
        db_path: Optional database path override
        
    Returns:
        Generated integer patient_id
    """
    query = """
    INSERT INTO patients (patient_name, gender, age, educ_level, ses, mmse, etiv, nwbv, asf)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
    """
    params = (
        str(data.get("patient_name", "Anonymous")),
        str(data.get("gender")),
        float(data.get("age")),
        int(data.get("educ_level")),
        float(data.get("ses")),
        float(data.get("mmse")),
        float(data.get("etiv")),
        float(data.get("nwbv")),
        float(data.get("asf"))
    )
    
    conn = get_connection(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute(query, params)
        conn.commit()
        patient_id = cursor.lastrowid
        return patient_id
    finally:
        conn.close()


def add_prediction(
    patient_id: int,
    prediction_label: str,
    probability: float,
    risk_level: str,
    model_name: str,
    db_path: Optional[str] = None
) -> int:
    """
    Inserts a new model prediction linked to an existing patient.
    
    Args:
        patient_id: Target patient ID foreign key
        prediction_label: Text diagnosis (e.g., 'No Dementia', 'Very Mild Dementia')
        probability: Diagnostic confidence score between 0.0 and 1.0
        risk_level: Clinical stage risk level (e.g., 'CDR 0.0', 'CDR 0.5', 'CDR 1.0+')
        model_name: Name of inference model
        db_path: Optional database path override
        
    Returns:
        Generated integer prediction_id
    """
    query = """
    INSERT INTO predictions (patient_id, prediction_label, probability, risk_level, model_name)
    VALUES (?, ?, ?, ?, ?);
    """
    params = (
        int(patient_id),
        str(prediction_label),
        float(probability),
        str(risk_level),
        str(model_name)
    )
    
    conn = get_connection(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute(query, params)
        conn.commit()
        prediction_id = cursor.lastrowid
        return prediction_id
    finally:
        conn.close()


def get_all_records(db_path: Optional[str] = None) -> pd.DataFrame:
    """
    Retrieves all patient records joined with their most recent prediction output.
    
    Args:
        db_path: Optional database path override
        
    Returns:
        pandas DataFrame of joined patient records and latest predictions
    """
    query = """
    SELECT 
        p.patient_id,
        p.patient_name,
        p.created_at,
        p.gender,
        p.age,
        p.educ_level,
        p.ses,
        p.mmse,
        p.etiv,
        p.nwbv,
        p.asf,
        pr.prediction_id,
        pr.prediction_label,
        pr.probability,
        pr.risk_level,
        pr.model_name,
        pr.predicted_at
    FROM patients p
    LEFT JOIN (
        SELECT pred1.*
        FROM predictions pred1
        INNER JOIN (
            SELECT patient_id, MAX(prediction_id) AS max_pred_id
            FROM predictions
            GROUP BY patient_id
        ) pred2 ON pred1.prediction_id = pred2.max_pred_id
    ) pr ON p.patient_id = pr.patient_id
    ORDER BY p.patient_id DESC;
    """
    conn = get_connection(db_path)
    try:
        df = pd.read_sql_query(query, conn)
        return df
    finally:
        conn.close()


def get_patient_history(patient_id: int, db_path: Optional[str] = None) -> pd.DataFrame:
    """
    Retrieves all historical predictions stored for a given patient.
    
    Args:
        patient_id: Integer patient identifier
        db_path: Optional database path override
        
    Returns:
        pandas DataFrame of predictions for the specified patient
    """
    query = """
    SELECT 
        prediction_id,
        patient_id,
        predicted_at,
        prediction_label,
        probability,
        risk_level,
        model_name
    FROM predictions
    WHERE patient_id = ?
    ORDER BY prediction_id DESC;
    """
    conn = get_connection(db_path)
    try:
        df = pd.read_sql_query(query, conn, params=(int(patient_id),))
        return df
    finally:
        conn.close()


def delete_patient(patient_id: int, db_path: Optional[str] = None) -> bool:
    """
    Deletes a patient record. Automatically cascades deletion to all associated predictions.
    
    Args:
        patient_id: Target patient ID to delete
        db_path: Optional database path override
        
    Returns:
        True if deletion succeeded and removed at least one row, False otherwise
    """
    query = "DELETE FROM patients WHERE patient_id = ?;"
    conn = get_connection(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute(query, (int(patient_id),))
        conn.commit()
        return cursor.rowcount > 0
    finally:
        conn.close()


def export_to_csv(path: str, db_path: Optional[str] = None) -> bool:
    """
    Exports all database records to a specified CSV file path.
    
    Args:
        path: Target CSV destination path
        db_path: Optional database path override
        
    Returns:
        True if export succeeded, False otherwise
    """
    df = get_all_records(db_path)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    df.to_csv(path, index=False)
    return True
