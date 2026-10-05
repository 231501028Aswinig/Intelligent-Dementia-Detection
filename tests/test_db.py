"""
Unit Tests for Database Module
Uses temporary file / in-memory SQLite database to test database initialization,
CRUD operations, CHECK constraints, and cascading deletions without affecting production.
"""

import os
import tempfile
import sqlite3
import unittest
import pandas as pd

from database.db import (
    get_connection,
    init_db,
    add_patient,
    add_prediction,
    get_all_records,
    get_patient_history,
    delete_patient,
    export_to_csv
)


class TestDatabaseModule(unittest.TestCase):
    def setUp(self):
        # Create a temporary file database for testing
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "test_patients.db")
        init_db(self.db_path)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_init_db_idempotent(self):
        """Test that init_db can be called multiple times without raising errors."""
        try:
            init_db(self.db_path)
            init_db(self.db_path)
        except Exception as e:
            self.fail(f"init_db raised an unexpected exception on repeated calls: {e}")

    def test_add_patient_and_add_prediction_round_trip(self):
        """Test inserting a patient and linking a prediction."""
        patient_data = {
            "patient_name": "Test Subject 01",
            "gender": "Female",
            "age": 74.0,
            "educ_level": 2,
            "ses": 3.0,
            "mmse": 29.0,
            "etiv": 1344.0,
            "nwbv": 0.743,
            "asf": 1.306
        }
        patient_id = add_patient(patient_data, db_path=self.db_path)
        self.assertIsInstance(patient_id, int)
        self.assertGreater(patient_id, 0)

        pred_id = add_prediction(
            patient_id=patient_id,
            prediction_label="No Dementia (Cognitively Normal)",
            probability=0.852,
            risk_level="STAGE: NORMAL (CDR 0.0)",
            model_name="RandomForestClassifier",
            db_path=self.db_path
        )
        self.assertIsInstance(pred_id, int)
        self.assertGreater(pred_id, 0)

        # Verify retrieval via get_all_records
        df = get_all_records(db_path=self.db_path)
        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]["patient_id"], patient_id)
        self.assertEqual(df.iloc[0]["prediction_label"], "No Dementia (Cognitively Normal)")
        self.assertAlmostEqual(df.iloc[0]["probability"], 0.852, places=3)

    def test_check_constraints_reject_invalid_data(self):
        """Test that sqlite CHECK constraints reject invalid ranges."""
        # 1. Invalid Age (< 18)
        invalid_age = {
            "patient_name": "Invalid Age",
            "gender": "Female",
            "age": 10.0,  # Below 18
            "educ_level": 2,
            "ses": 3.0,
            "mmse": 29.0,
            "etiv": 1344.0,
            "nwbv": 0.743,
            "asf": 1.306
        }
        with self.assertRaises(sqlite3.IntegrityError):
            add_patient(invalid_age, db_path=self.db_path)

        # 2. Invalid MMSE (> 30)
        invalid_mmse = {
            "patient_name": "Invalid MMSE",
            "gender": "Male",
            "age": 70.0,
            "educ_level": 3,
            "ses": 2.0,
            "mmse": 35.0,  # Above 30
            "etiv": 1400.0,
            "nwbv": 0.720,
            "asf": 1.200
        }
        with self.assertRaises(sqlite3.IntegrityError):
            add_patient(invalid_mmse, db_path=self.db_path)

        # 3. Invalid Gender (not in Female, Male)
        invalid_gender = {
            "patient_name": "Invalid Gender",
            "gender": "Unknown",
            "age": 70.0,
            "educ_level": 3,
            "ses": 2.0,
            "mmse": 28.0,
            "etiv": 1400.0,
            "nwbv": 0.720,
            "asf": 1.200
        }
        with self.assertRaises(sqlite3.IntegrityError):
            add_patient(invalid_gender, db_path=self.db_path)

        # 4. Invalid Probability (> 1.0)
        valid_patient = {
            "patient_name": "Valid Patient",
            "gender": "Male",
            "age": 70.0,
            "educ_level": 3,
            "ses": 2.0,
            "mmse": 28.0,
            "etiv": 1400.0,
            "nwbv": 0.720,
            "asf": 1.200
        }
        pid = add_patient(valid_patient, db_path=self.db_path)
        with self.assertRaises(sqlite3.IntegrityError):
            add_prediction(
                patient_id=pid,
                prediction_label="Test",
                probability=1.5,  # Above 1.0
                risk_level="Test",
                model_name="TestModel",
                db_path=self.db_path
            )

    def test_cascade_deletion(self):
        """Test that deleting a patient cascades and removes linked predictions."""
        patient_data = {
            "patient_name": "Cascade Patient",
            "gender": "Male",
            "age": 68.0,
            "educ_level": 4,
            "ses": 2.0,
            "mmse": 27.0,
            "etiv": 1500.0,
            "nwbv": 0.730,
            "asf": 1.180
        }
        pid = add_patient(patient_data, db_path=self.db_path)
        add_prediction(pid, "Pred 1", 0.75, "Stage 1", "Model A", db_path=self.db_path)
        add_prediction(pid, "Pred 2", 0.80, "Stage 2", "Model A", db_path=self.db_path)

        history_before = get_patient_history(pid, db_path=self.db_path)
        self.assertEqual(len(history_before), 2)

        # Delete patient
        deleted = delete_patient(pid, db_path=self.db_path)
        self.assertTrue(deleted)

        # Verify patient is gone
        all_records = get_all_records(db_path=self.db_path)
        self.assertEqual(len(all_records), 0)

        # Verify cascade removed predictions
        conn = get_connection(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM predictions WHERE patient_id = ?;", (pid,))
        pred_count = cursor.fetchone()[0]
        conn.close()
        self.assertEqual(pred_count, 0)

    def test_get_patient_history_and_csv_export(self):
        """Test retrieving multiple predictions for a patient and CSV export."""
        patient_data = {
            "patient_name": "History Patient",
            "gender": "Female",
            "age": 80.0,
            "educ_level": 3,
            "ses": 3.0,
            "mmse": 22.0,
            "etiv": 1420.0,
            "nwbv": 0.690,
            "asf": 1.250
        }
        pid = add_patient(patient_data, db_path=self.db_path)
        add_prediction(pid, "Stage 0.5", 0.70, "Mild", "Model A", db_path=self.db_path)
        add_prediction(pid, "Stage 1.0", 0.92, "Dementia", "Model B", db_path=self.db_path)

        history = get_patient_history(pid, db_path=self.db_path)
        self.assertEqual(len(history), 2)

        # Test CSV Export
        csv_path = os.path.join(self.temp_dir.name, "exported_records.csv")
        success = export_to_csv(csv_path, db_path=self.db_path)
        self.assertTrue(success)
        self.assertTrue(os.path.exists(csv_path))
        df_csv = pd.read_csv(csv_path)
        self.assertEqual(len(df_csv), 1)


if __name__ == "__main__":
    unittest.main()
