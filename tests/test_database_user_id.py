import os
import sqlite3
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.database import DatabaseManager, LogType


class DatabaseUserIdTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory(prefix="db-user-id-")
        self.db_path = os.path.join(self.temp_dir.name, "users.db")
        self.db = DatabaseManager(self.db_path)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_registration_login_logs_and_face_vector_use_user_id(self):
        vector = np.arange(128, dtype=np.float32)
        success, message = self.db.register_user(
            "alice", "pass1234", "pass1234", vector
        )
        self.assertTrue(success, message)

        success, message, user_id = self.db.check_user("alice", "pass1234")
        self.assertTrue(success, message)
        self.assertEqual(len(user_id), 64)
        self.assertEqual(len(self.db.get_user_vector(user_id)), 128)
        self.assertTrue(self.db.get_logs(user_id))

        conn = sqlite3.connect(self.db_path)
        try:
            stored_user_id = conn.execute(
                "SELECT id FROM users WHERE username = 'alice'"
            ).fetchone()[0]
            log_user_id = conn.execute(
                "SELECT user_id FROM logs ORDER BY id DESC LIMIT 1"
            ).fetchone()[0]
            mapping_user_id = conn.execute(
                "SELECT user_id FROM user_vec_mapping"
            ).fetchone()[0]
        finally:
            conn.close()
        self.assertEqual(user_id, stored_user_id)
        self.assertEqual(log_user_id, user_id)
        self.assertEqual(mapping_user_id, user_id)

    def test_migrates_legacy_username_references(self):
        legacy_db_path = os.path.join(self.temp_dir.name, "old.db")
        conn = sqlite3.connect(legacy_db_path)
        conn.executescript("""
            CREATE TABLE users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE,
                password TEXT NOT NULL,
                created_at DATETIME DEFAULT (datetime('now','localtime'))
            );
            CREATE TABLE logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL,
                log_type TEXT NOT NULL,
                datetime DATETIME DEFAULT (datetime('now','localtime'))
            );
            CREATE TABLE user_vec_mapping (
                vector_id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE
            );
            INSERT INTO users (username, password) VALUES ('alice', 'stored-hash');
            INSERT INTO logs (username, log_type) VALUES ('alice', 'login');
            INSERT INTO user_vec_mapping (vector_id, username) VALUES (7, 'alice');
        """)
        conn.commit()
        conn.close()

        migrated_db = DatabaseManager(legacy_db_path)
        conn = sqlite3.connect(legacy_db_path)
        try:
            user_id = conn.execute(
                "SELECT id FROM users WHERE username = 'alice'"
            ).fetchone()[0]
            migrated_mapping = conn.execute(
                "SELECT vector_id, user_id FROM user_vec_mapping"
            ).fetchone()
        finally:
            conn.close()
        self.assertEqual(migrated_db.get_logs(user_id)[0][1], "alice")
        self.assertEqual(migrated_mapping, (7, user_id))

        conn = sqlite3.connect(legacy_db_path)
        try:
            user_columns = {
                row[1]: row[2].upper()
                for row in conn.execute("PRAGMA table_info(users)")
            }
            log_columns = {
                row[1] for row in conn.execute("PRAGMA table_info(logs)")
            }
            self.assertEqual(user_columns["id"], "TEXT")
            self.assertIn("user_id", log_columns)
        finally:
            conn.close()


if __name__ == "__main__":
    unittest.main()