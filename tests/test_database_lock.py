import os
import sqlite3
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.database import DatabaseManager


class DatabaseLockTest(unittest.TestCase):
    def test_register_user_waits_for_ongoing_transaction(self):
        temp_dir = tempfile.mkdtemp(prefix="db-lock-")
        db_path = os.path.join(temp_dir, "users.db")
        db = DatabaseManager(db_path)

        started = threading.Event()

        def hold_lock():
            conn = sqlite3.connect(db_path, timeout=30)
            try:
                conn.execute("BEGIN IMMEDIATE")
                conn.execute(
                    "INSERT INTO users (id, username, password) VALUES (?, ?, ?)",
                    (db.create_user_id(), "lock_holder", "hashed"),
                )
                started.set()
                time.sleep(8)
                conn.commit()
            finally:
                conn.close()

        lock_thread = threading.Thread(target=hold_lock)
        lock_thread.start()
        started.wait(2)

        success, message = db.register_user("newuser", "pass1234", "pass1234")

        lock_thread.join()

        self.assertTrue(success, message)


if __name__ == "__main__":
    unittest.main()
