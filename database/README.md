# SQLite Database Module: Intelligent Dementia Detection System

This module provides a lightweight, local SQLite database for persisting patient parameters and machine learning diagnostic predictions.

## Architecture

- **Engine:** SQLite 3 (Python standard library `sqlite3`)
- **Default Database Location:** `database/patients.db` (Excluded from git tracking for privacy)
- **Schema Definition:** `database/schema.sql`

## Tables

### 1. `patients`
Stores demographic, cognitive, and volumetric anatomical indices for each patient.
- `patient_id` (INTEGER PRIMARY KEY AUTOINCREMENT)
- `created_at` (TIMESTAMP DEFAULT CURRENT_TIMESTAMP)
- `patient_name` (TEXT)
- `gender` (TEXT CHECK ('Female', 'Male'))
- `age` (REAL CHECK (age >= 18 AND age <= 105))
- `educ_level` (INTEGER CHECK (educ_level >= 1 AND educ_level <= 5))
- `ses` (REAL CHECK (ses >= 1.0 AND ses <= 5.0))
- `mmse` (REAL CHECK (mmse >= 0.0 AND mmse <= 30.0))
- `etiv` (REAL CHECK (etiv >= 800.0 AND etiv <= 2500.0))
- `nwbv` (REAL CHECK (nwbv >= 0.500 AND nwbv <= 0.950))
- `asf` (REAL CHECK (asf >= 0.700 AND asf <= 1.800))

### 2. `predictions`
Stores model diagnostic predictions linked to a specific patient.
- `prediction_id` (INTEGER PRIMARY KEY AUTOINCREMENT)
- `patient_id` (INTEGER NOT NULL, FOREIGN KEY REFERENCES `patients(patient_id)` ON DELETE CASCADE)
- `predicted_at` (TIMESTAMP DEFAULT CURRENT_TIMESTAMP)
- `prediction_label` (TEXT NOT NULL)
- `probability` (REAL NOT NULL CHECK (probability >= 0.0 AND probability <= 1.0))
- `risk_level` (TEXT)
- `model_name` (TEXT NOT NULL)

## Core Functions (`database/db.py`)

- `get_connection(db_path=None)`: Returns an active connection with foreign keys enabled and `sqlite3.Row` factory.
- `init_db(db_path=None)`: Idempotently executes `schema.sql` to initialize tables and indexes.
- `add_patient(data: dict, db_path=None)`: Inserts a patient record and returns the assigned `patient_id`.
- `add_prediction(patient_id, prediction_label, probability, risk_level, model_name, db_path=None)`: Stores a model prediction.
- `get_all_records(db_path=None)`: Returns a DataFrame containing all patients joined with their latest prediction.
- `get_patient_history(patient_id, db_path=None)`: Returns all prediction records for a given patient.
- `delete_patient(patient_id, db_path=None)`: Deletes a patient and cascades deletion to all linked predictions.
- `export_to_csv(path, db_path=None)`: Exports current records to a CSV file.

## Privacy & Security

- Patient data stored in SQLite is strictly for local clinical workflow demonstration.
- `database/patients.db` and all `*.db` files are included in `.gitignore` to prevent committing patient data to version control.
