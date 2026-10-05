-- =============================================================================
-- Database Schema for Intelligent Dementia Detection System
-- Engine: SQLite 3
-- =============================================================================

-- Enable Foreign Key enforcement
PRAGMA foreign_keys = ON;

-- -----------------------------------------------------------------------------
-- Table: patients
-- Stores patient demographic, cognitive, and volumetric anatomical parameters
-- -----------------------------------------------------------------------------
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

-- -----------------------------------------------------------------------------
-- Table: predictions
-- Stores model prediction outputs and diagnostic confidence linked to a patient
-- -----------------------------------------------------------------------------
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

-- -----------------------------------------------------------------------------
-- Index for efficient join and history retrieval
-- -----------------------------------------------------------------------------
CREATE INDEX IF NOT EXISTS idx_predictions_patient_id ON predictions(patient_id);
