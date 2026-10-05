"""
Seed Script for Intelligent Dementia Detection System
Populates exactly 5 synthetic demonstration patient records with associated predictions.
All records are strictly synthetic/mock data for demonstration and testing purposes.
"""

import os
from database.db import init_db, add_patient, add_prediction, get_all_records

DEMO_PATIENTS = [
    {
        "patient": {
            "patient_name": "DEMO_PATIENT_001 (Synthetic Healthy)",
            "gender": "Female",
            "age": 72.0,
            "educ_level": 3,
            "ses": 2.0,
            "mmse": 29.0,
            "etiv": 1380.0,
            "nwbv": 0.760,
            "asf": 1.272
        },
        "prediction": {
            "prediction_label": "No Dementia (Cognitively Normal)",
            "probability": 0.885,
            "risk_level": "STAGE: NORMAL (CDR 0.0)",
            "model_name": "RandomForest 3-Class Pipeline"
        }
    },
    {
        "patient": {
            "patient_name": "DEMO_PATIENT_002 (Synthetic Mild)",
            "gender": "Male",
            "age": 76.0,
            "educ_level": 2,
            "ses": 3.0,
            "mmse": 26.0,
            "etiv": 1490.0,
            "nwbv": 0.710,
            "asf": 1.178
        },
        "prediction": {
            "prediction_label": "Very Mild Dementia (Prodromal Stage)",
            "probability": 0.712,
            "risk_level": "STAGE: VERY MILD (CDR 0.5)",
            "model_name": "RandomForest 3-Class Pipeline"
        }
    },
    {
        "patient": {
            "patient_name": "DEMO_PATIENT_003 (Synthetic Dementia)",
            "gender": "Female",
            "age": 81.0,
            "educ_level": 1,
            "ses": 4.0,
            "mmse": 19.0,
            "etiv": 1420.0,
            "nwbv": 0.672,
            "asf": 1.235
        },
        "prediction": {
            "prediction_label": "Clinical Dementia (Established Stage)",
            "probability": 0.941,
            "risk_level": "STAGE: DEMENTIA (CDR 1.0+)",
            "model_name": "RandomForest 3-Class Pipeline"
        }
    },
    {
        "patient": {
            "patient_name": "DEMO_PATIENT_004 (Synthetic Borderline)",
            "gender": "Male",
            "age": 69.0,
            "educ_level": 4,
            "ses": 2.0,
            "mmse": 28.0,
            "etiv": 1560.0,
            "nwbv": 0.735,
            "asf": 1.125
        },
        "prediction": {
            "prediction_label": "No Dementia (Cognitively Normal)",
            "probability": 0.645,
            "risk_level": "STAGE: NORMAL (CDR 0.0)",
            "model_name": "RandomForest 3-Class Pipeline"
        }
    },
    {
        "patient": {
            "patient_name": "DEMO_PATIENT_005 (Synthetic Advanced)",
            "gender": "Male",
            "age": 85.0,
            "educ_level": 2,
            "ses": 3.0,
            "mmse": 15.0,
            "etiv": 1610.0,
            "nwbv": 0.655,
            "asf": 1.090
        },
        "prediction": {
            "prediction_label": "Clinical Dementia (Established Stage)",
            "probability": 0.978,
            "risk_level": "STAGE: DEMENTIA (CDR 1.0+)",
            "model_name": "RandomForest 3-Class Pipeline"
        }
    }
]


def seed_database(db_path=None):
    """
    Initializes database and inserts 5 mock demonstration records.
    """
    print(f"Initializing database schema...")
    init_db(db_path)
    
    print(f"Seeding exactly {len(DEMO_PATIENTS)} synthetic demo patients...")
    for idx, item in enumerate(DEMO_PATIENTS, 1):
        p_data = item["patient"]
        pred_data = item["prediction"]
        
        patient_id = add_patient(p_data, db_path=db_path)
        pred_id = add_prediction(
            patient_id=patient_id,
            prediction_label=pred_data["prediction_label"],
            probability=pred_data["probability"],
            risk_level=pred_data["risk_level"],
            model_name=pred_data["model_name"],
            db_path=db_path
        )
        print(f"  [Seeded #{idx}] Patient ID: {patient_id} ({p_data['patient_name']}) -> Prediction ID: {pred_id}")
        
    records = get_all_records(db_path)
    print(f"Successfully seeded database. Total records: {len(records)}")
    return records


if __name__ == "__main__":
    seed_database()
