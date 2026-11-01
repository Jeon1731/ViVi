import sqlite3
import sqlite_vec
import struct
import hashlib
import os
from pathlib import Path
from enum import Enum


class LogType(Enum):
    """로그 타입 정의"""
    LOGIN = "login"                     # 로그인
    LOGOUT = "logout"                   # 로그아웃
    LOCK = "lock"                       # 화면잠김


class DatabaseManager:
    def __init__(self, db_path="data/database.db"):
        """데이터베이스 초기화"""
        self.db_path = db_path
        
        # data 디렉토리 생성
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        
        self.create_tables()

    def get_connection(self):
        """데이터베이스 연결"""
        conn = sqlite3.connect(
            self.db_path,
            timeout=30,
            check_same_thread=False,
        )
        conn.enable_load_extension(True)
        try:
            sqlite_vec.load(conn)
        finally:
            conn.enable_load_extension(False)
        conn.execute("PRAGMA busy_timeout = 30000")
        conn.execute("PRAGMA journal_mode = WAL")
        return conn

    def create_tables(self):
        """회원 테이블 생성"""
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            tables = {
                row[0]
                for row in cursor.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'"
                )
            }
            if "users" in tables:
                columns = {
                    row[1]: row[2].upper()
                    for row in cursor.execute("PRAGMA table_info(users)")
                }
                legacy_schema = columns.get("id") != "TEXT"
                for table in ("logs", "user_vec_mapping"):
                    if table in tables:
                        table_columns = {
                            row[1]
                            for row in cursor.execute(f"PRAGMA table_info({table})")
                        }
                        legacy_schema |= "user_id" not in table_columns
                if legacy_schema:
                    self._migrate_legacy_schema(conn, tables)

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    id TEXT PRIMARY KEY NOT NULL,
                    username TEXT NOT NULL UNIQUE,
                    password TEXT NOT NULL,
                    created_at DATETIME DEFAULT (datetime('now','localtime'))
                )
            ''')
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    log_type TEXT NOT NULL,
                    datetime DATETIME DEFAULT (datetime('now','localtime')),
                    FOREIGN KEY (user_id) REFERENCES users(id)
                )
            ''')
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS user_vec_mapping (
                    vector_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL UNIQUE,
                    FOREIGN KEY (user_id) REFERENCES users(id)
                )
            """)
            cursor.execute('''
                CREATE VIRTUAL TABLE IF NOT EXISTS vectors USING vec0(
                    embedding float[128]
                )
            ''')
            conn.commit()
        finally:
            conn.close()

    @staticmethod
    def create_user_id():
        """난수 기반 SHA-256 사용자 ID 생성"""
        return hashlib.sha256(os.urandom(32)).hexdigest()

    def _migrate_legacy_schema(self, conn, tables):
        """username을 참조하던 기존 테이블을 user_id 기반으로 변환"""
        cursor = conn.cursor()
        users = cursor.execute(
            "SELECT username, password, created_at FROM users"
        ).fetchall()
        user_ids = {
            username: self.create_user_id()
            for username, _, _ in users
        }

        logs = cursor.execute(
            "SELECT id, username, log_type, datetime FROM logs"
        ).fetchall() if "logs" in tables else []
        mappings = cursor.execute(
            "SELECT vector_id, username FROM user_vec_mapping"
        ).fetchall() if "user_vec_mapping" in tables else []

        cursor.execute("DROP TABLE IF EXISTS __vivi_users")
        cursor.execute("DROP TABLE IF EXISTS __vivi_logs")
        cursor.execute("DROP TABLE IF EXISTS __vivi_user_vec_mapping")
        cursor.execute('''
            CREATE TABLE __vivi_users (
                id TEXT PRIMARY KEY NOT NULL,
                username TEXT NOT NULL UNIQUE,
                password TEXT NOT NULL,
                created_at DATETIME DEFAULT (datetime('now','localtime'))
            )
        ''')
        cursor.execute('''
            CREATE TABLE __vivi_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                log_type TEXT NOT NULL,
                datetime DATETIME DEFAULT (datetime('now','localtime')),
                FOREIGN KEY (user_id) REFERENCES __vivi_users(id)
            )
        ''')
        cursor.execute('''
            CREATE TABLE __vivi_user_vec_mapping (
                vector_id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL UNIQUE,
                FOREIGN KEY (user_id) REFERENCES __vivi_users(id)
            )
        ''')
        cursor.executemany(
            "INSERT INTO __vivi_users (id, username, password, created_at) VALUES (?, ?, ?, ?)",
            [
                (user_ids[username], username, password, created_at)
                for username, password, created_at in users
            ],
        )
        cursor.executemany(
            "INSERT INTO __vivi_logs (id, user_id, log_type, datetime) VALUES (?, ?, ?, ?)",
            [
                (log_id, user_ids[username], log_type, logged_at)
                for log_id, username, log_type, logged_at in logs
                if username in user_ids
            ],
        )
        cursor.executemany(
            "INSERT INTO __vivi_user_vec_mapping (vector_id, user_id) VALUES (?, ?)",
            [
                (vector_id, user_ids[username])
                for vector_id, username in mappings
                if username in user_ids
            ],
        )

        for table in ("user_vec_mapping", "logs", "users"):
            if table in tables:
                cursor.execute(f"DROP TABLE {table}")
        cursor.execute("ALTER TABLE __vivi_users RENAME TO users")
        cursor.execute("ALTER TABLE __vivi_logs RENAME TO logs")
        cursor.execute("ALTER TABLE __vivi_user_vec_mapping RENAME TO user_vec_mapping")

    def hash_password(self, password):
        """비밀번호 해싱 (SHA-256)"""
        return hashlib.sha256(password.encode()).hexdigest()

    def register_user(self, username, password, password_confirm, face_vector=None):
        """회원 등록"""
        conn = None
        try:
            # 비밀번호 일치 여부 확인
            if password != password_confirm:
                return False, "비밀번호가 일치하지 않습니다."

            # 비밀번호 길이 확인
            if len(password) < 4:
                return False, "비밀번호는 4자 이상이어야 합니다."

            # 아이디 길이 확인
            if len(username) < 3:
                return False, "아이디는 3자 이상이어야 합니다."

            conn = self.get_connection()
            cursor = conn.cursor()

            # 비밀번호 해싱
            hashed_password = self.hash_password(password)
            user_id = self.create_user_id()

            # 사용자 추가
            cursor.execute('''
                INSERT INTO users (id, username, password)
                VALUES (?, ?, ?)
            ''', (user_id, username, hashed_password))

            if face_vector is not None:
                success, message = self.mapping_user_and_vector(user_id, face_vector, conn)
                if not success:
                    conn.rollback()
                    return False, message

            conn.commit()
            return True, "회원 등록 성공!"

        except sqlite3.IntegrityError:
            return False, "이미 존재하는 아이디입니다."
        except Exception as e:
            return False, f"오류 발생: {str(e)}"
        finally:
            if conn is not None:
                conn.close()

    def check_user(self, username, password):
        """로그인 확인"""
        conn = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor()

            hashed_password = self.hash_password(password)

            cursor.execute('''
                SELECT id FROM users
                WHERE username = ? AND password = ?
            ''', (username, hashed_password))

            user = cursor.fetchone()

            if user:
                user_id = user[0]
                self.log_event(user_id, LogType.LOGIN)
                return True, "로그인 성공!", user_id
            else:
                return False, "아이디 또는 비밀번호가 잘못되었습니다.", None

        except Exception as e:
            return False, f"오류 발생: {str(e)}", None
        finally:
            if conn is not None:
                conn.close()

    def user_exists(self, username):
        """아이디 중복 확인"""
        conn = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor()

            cursor.execute('SELECT id FROM users WHERE username = ?', (username,))
            user = cursor.fetchone()

            return user is not None

        except Exception as e:
            print(f"오류 발생: {str(e)}")
            return False
        finally:
            if conn is not None:
                conn.close()

    def log_event(self, user_id, log_type):
        """로그 기록"""
        conn = None
        try:
            if not isinstance(log_type, LogType):
                return False, "유효하지 않은 로그 타입입니다."

            conn = self.get_connection()
            cursor = conn.cursor()

            cursor.execute('''
                INSERT INTO logs (user_id, log_type)
                VALUES (?, ?)
            ''', (user_id, log_type.value))

            conn.commit()
            return True, "로그 기록 성공"

        except Exception as e:
            return False, f"로그 기록 오류: {str(e)}"
        finally:
            if conn is not None:
                conn.close()

    def get_logs(self, user_id=None, limit=10):
        """로그 조회"""
        conn = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor()

            if user_id:
                cursor.execute('''
                    SELECT logs.id, users.username, logs.log_type, logs.datetime
                    FROM logs
                    JOIN users ON users.id = logs.user_id
                    WHERE logs.user_id = ?
                    ORDER BY logs.datetime DESC
                    LIMIT ?
                ''', (user_id, limit))
            else:
                cursor.execute('''
                    SELECT logs.id, users.username, logs.log_type, logs.datetime
                    FROM logs
                    JOIN users ON users.id = logs.user_id
                    ORDER BY logs.datetime DESC
                    LIMIT ?
                ''', (limit,))

            logs = cursor.fetchall()
            return logs

        except Exception as e:
            print(f"로그 조회 오류: {str(e)}")
            return []
        finally:
            if conn is not None:
                conn.close()

    def serialize_vector(self, vector):
        """벡터를 바이트(Blob)로 변환"""
        values = vector.reshape(-1).tolist()
        return struct.pack(f"{len(values)}f", *(float(value) for value in values))

    def get_user_vector(self, user_id):
        """사용자 ID에 연결된 얼굴 벡터 조회"""
        conn = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            cursor.execute("""
                SELECT v.embedding
                FROM user_vec_mapping AS m
                JOIN vectors AS v ON v.rowid = m.vector_id
                WHERE m.user_id = ?
            """, (user_id,))
            row = cursor.fetchone()

            if row is None:
                return None

            return struct.unpack("128f", row[0])
        except Exception as e:
            print(f"얼굴 벡터 조회 오류: {str(e)}")
            return None
        finally:
            if conn is not None:
                conn.close()

    def mapping_user_and_vector(self, user_id, vector, conn=None):
        """매핑 테이블 삽입"""
        close_conn = conn is None
        try:
            if conn is None:
                conn = self.get_connection()
            cursor = conn.cursor()

            cursor.execute("INSERT INTO user_vec_mapping (user_id) VALUES (?)", (user_id,))
            vector_id = cursor.lastrowid

            cursor.execute("INSERT INTO vectors (rowid, embedding) VALUES (?, ?)", (vector_id, self.serialize_vector(vector)))

            if close_conn:
                conn.commit()
            return True, "얼굴 등록 성공!"
        except Exception as e:
            return False, f"매핑 테이블 오류: {str(e)}"
        finally:
            if close_conn and conn is not None:
                conn.close()